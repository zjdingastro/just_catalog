#!/usr/bin/env python
# coding: utf-8

import os, glob
import numpy as np
import healpy as hp
from astropy.table import Table, vstack, join
import matplotlib.pyplot as plt
from multiprocessing import get_context
import argparse


def process_one_fba_file(infile):
    """Read one tile FITS and build assigned table for that tile."""
    cat_assigned = Table.read(infile, hdu=1)

    tile_id = int(os.path.basename(infile).split("_")[-1].split(".")[0])
    tile_id64 = np.int64(tile_id)
    assigned_fiber = np.asarray(cat_assigned["FIBERID"], dtype=np.int64)

    cat_assigned["TILEID"] = tile_id64
    cat_assigned["TILELOCID"] = tile_id64 * np.int64(10000) + assigned_fiber

    return tile_id, cat_assigned


def _target_tiles_fibers_for_block(target_id, tile_block, fiber_block):
    """Format TILES and FIBERS strings for one TARGETID block."""
    if tile_block.size == 1:
        return (
            int(target_id),
            str(int(tile_block[0])),
            str(int(fiber_block[0])),
        )
    pair_change = np.r_[
        True,
        (tile_block[1:] != tile_block[:-1]) | (fiber_block[1:] != fiber_block[:-1]),
    ]
    tiles_u = tile_block[pair_change]
    fibers_u = fiber_block[pair_change]
    return (
        int(target_id),
        "-".join(map(str, tiles_u.tolist())),
        "-".join(map(str, fibers_u.tolist())),
    )


def _aggregate_assigned_group_chunk(t_sorted, tile_sorted, fiber_sorted, group_starts, group_ends, i0, i1):
    rows = []
    for gi in range(i0, i1):
        s = int(group_starts[gi])
        e = int(group_ends[gi])
        rows.append(
            _target_tiles_fibers_for_block(
                t_sorted[s], tile_sorted[s:e], fiber_sorted[s:e]
            )
        )
    return rows


def aggregate_assigned_by_target(cat_assigned_dict, n_workers=1, tiles_dtype="U64", fibers_dtype="U64"):
    """One row per TARGETID with TILES and FIBERS hyphen lists in matching order."""
    all_assigned = vstack(
        [cat_assigned_dict[k]["TARGETID", "FIBERID", "TILEID"] for k in cat_assigned_dict]
    )

    target_ids = np.asarray(all_assigned["TARGETID"])
    tile_ids = np.asarray(all_assigned["TILEID"], dtype=np.int64)
    fiber_ids = np.asarray(all_assigned["FIBERID"], dtype=np.int64)

    order = np.lexsort((fiber_ids, tile_ids, target_ids))
    t_sorted = target_ids[order]
    tile_sorted = tile_ids[order]
    fiber_sorted = fiber_ids[order]

    n = len(t_sorted)
    group_starts = np.r_[0, 1 + np.flatnonzero(t_sorted[1:] != t_sorted[:-1])]
    group_ends = np.r_[group_starts[1:], n]
    n_groups = len(group_starts)

    if n_workers <= 1:
        rows = _aggregate_assigned_group_chunk(
            t_sorted, tile_sorted, fiber_sorted, group_starts, group_ends, 0, n_groups
        )
    else:
        ctx = get_context("fork")
        chunk_groups = max(1, n_groups // (n_workers * 4))
        tasks = []
        for i0 in range(0, n_groups, chunk_groups):
            i1 = min(n_groups, i0 + chunk_groups)
            tasks.append((t_sorted, tile_sorted, fiber_sorted, group_starts, group_ends, i0, i1))
        rows = []
        with ctx.Pool(processes=n_workers) as pool:
            for chunk in pool.starmap(_aggregate_assigned_group_chunk, tasks):
                rows.extend(chunk)

    target_out, tiles_out, fibers_out = zip(*rows)
    return Table(
        {
            "TARGETID": np.asarray(target_out, dtype=np.int64),
            "TILES": np.array(tiles_out, dtype=tiles_dtype),
            "FIBERS": np.array(fibers_out, dtype=fibers_dtype),
        }
    )


def count_targets_per_healpix_pixel(ra_deg, dec_deg, nside, nest=False):
    """Return HEALPix pixel ID per target and total target count in each pixel."""
    ra_deg = np.asarray(ra_deg, dtype=np.float64)
    dec_deg = np.asarray(dec_deg, dtype=np.float64)
    pix = hp.ang2pix(nside, ra_deg, dec_deg, lonlat=True, nest=nest).astype(np.int64)
    npix = hp.nside2npix(nside)
    return pix, np.bincount(pix, minlength=npix)


def lookup_healpix_for_targetids(target_ids, parent_ids, parent_pix):
    """Map output TARGETIDs to parent HEALPix pixel IDs."""
    target_ids = np.asarray(target_ids)
    order = np.argsort(parent_ids)
    sorted_ids = parent_ids[order]
    sorted_pix = parent_pix[order]
    pos = np.searchsorted(sorted_ids, target_ids)
    if not np.all(sorted_ids[pos] == target_ids):
        missing = target_ids[sorted_ids[pos] != target_ids]
        raise ValueError(f"{len(missing)} TARGETIDs missing from parent catalog")
    return sorted_pix[pos]


def add_completeness_weight(output, cat_parent, nside, nest=False, weight_col="COMP_FA"):
    """
    Add A/B = fiber assigned completeness per HEALPix pixel, where
      A = number of fiber-assigned output targets in the pixel
      B = number of parent-catalog targets in the pixel 
    """
    parent_ids = np.asarray(cat_parent["idx"])
    pix_parent, count_parent = count_targets_per_healpix_pixel(
        cat_parent["ra"], cat_parent["dec"], nside, nest=nest
    )
    pix_output = lookup_healpix_for_targetids(output["TARGETID"], parent_ids, pix_parent)

    npix = hp.nside2npix(nside)
    count_fa = np.bincount(pix_output, minlength=npix)

    count_parent_row = count_parent[pix_output]
    count_fa_row = count_fa[pix_output]
    if np.any(count_parent_row == 0):
        raise ValueError("Found parent targets in pixels with zero count")

    output[weight_col] = count_fa_row.astype(np.float64) / count_parent_row.astype(np.float64)
    return output


def build_completeness_map(count_fa, count_parent, nside):
    """Build per-pixel completeness = n_assigned / n_parent on the parent footprint."""
    npix = hp.nside2npix(nside)
    weight_map = np.full(npix, hp.UNSEEN, dtype=np.float64)

    in_parent = count_parent > 0
    weight_map[in_parent] = 0.0
    assigned = count_fa > 0
    weight_map[assigned] = (
        count_fa[assigned].astype(np.float64) / count_parent[assigned].astype(np.float64)
    )
    return weight_map, in_parent, assigned


def _clip_cartview_ranges(ra_range, dec_range, pad_deg=1.0):
    """Clip lonra/latra to ranges accepted by healpy cartview."""
    ra0 = float(ra_range[0]) - pad_deg
    ra1 = float(ra_range[1]) + pad_deg
    dec0 = max(-90.0, float(dec_range[0]) - pad_deg)
    dec1 = min(90.0, float(dec_range[1]) + pad_deg)
    if dec0 >= dec1:
        dec1 = min(90.0, dec0 + 0.5)
    if ra0 >= ra1:
        ra1 = ra0 + 0.5
    return (ra0, ra1), (dec0, dec1)


def plot_completeness_map(
    count_fa,
    count_parent,
    nside,
    nest=False,
    n_passes=3,
    ra_range=(0.0, 90.0),
    dec_range=(0.0, 90.0),
    outfile=None,
):
    """Cartview + histogram for fiber assignment completeness on the parent footprint."""
    weight_map, in_parent, assigned = build_completeness_map(
        count_fa, count_parent, nside
    )
    parent_weights = weight_map[in_parent]
    assigned_weights = weight_map[assigned]

    vmin, vmax = 0.0, 1.0
    p2, p50, p98 = np.percentile(assigned_weights, [2, 50, 98])
    ra_range, dec_range = _clip_cartview_ranges(ra_range, dec_range)

    fig = plt.figure(figsize=(13, 10))
    hp.cartview(
        weight_map,
        nest=nest,
        coord="C",
        lonra=[ra_range[0], ra_range[1]],
        latra=[dec_range[0], dec_range[1]],
        min=vmin,
        max=vmax,
        title=f"Fiber assignment completeness ({n_passes} passes, HEALPix nside={nside})",
        unit="n_assigned / n_parent",
        cmap=plt.cm.RdYlGn,
        flip="astro",
        cbar=True,
        sub=(2, 1, 1),
    )
    hp.graticule(dmer=15, dpar=15, verbose=False)

    ax_hist = fig.add_subplot(2, 1, 2)
    ax_hist.hist(
        assigned_weights,
        bins=50,
        range=(0.0, 1.0),
        color="steelblue",
        edgecolor="white",
        linewidth=0.4,
    )
    ax_hist.axvline(p50, color="crimson", ls="--", lw=1.5, label=f"median={p50:.3f}")
    ax_hist.axvline(np.mean(assigned_weights), color="black", ls=":", lw=1.5,
                    label=f"mean={np.mean(assigned_weights):.3f}")
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.set_xlabel("Completeness per HEALPix pixel")
    ax_hist.set_ylabel("Number of pixels")
    ax_hist.set_title("Distribution on pixels with at least one assignment")
    ax_hist.legend(loc="upper left")

    summary = (
        f"parent footprint pixels={in_parent.sum():,}; "
        f"assigned pixels={assigned.sum():,}; "
        f"completeness min={assigned_weights.min():.3f}, "
        f"max={assigned_weights.max():.3f}, "
        f"p2={p2:.3f}, p98={p98:.3f}"
    )
    fig.text(0.08, 0.02, summary, ha="left", va="bottom", fontsize=10)
    fig.subplots_adjust(bottom=0.08, hspace=0.25)
    if outfile is not None:
        plt.savefig(outfile, dpi=150)
        print(f"saved completeness plot: {outfile}")
    else:
        plt.show()
    plt.close(fig)

    return weight_map


def main():
    parser = argparse.ArgumentParser(description="Make large-scale structure catalogs from fiber assigned targets")
    parser.add_argument("--mock_version", type=str, default="v1", help="Mock version")
    parser.add_argument("--Npasses", type=int, default=3, help="Number of passes")
    parser.add_argument("--seed", type=int, default=100, help="Random seed for the fiber assignment")
    parser.add_argument("--MTL_path", type=str, default="/home/zjding/fiberassignment/JUST/BGS_mock/Junyu_mock/data/v1/lightcone_ra_0_90_dec_0_90_rmagcut20.5.fits", help="Parent catalog file")
    parser.add_argument("--nside", type=int, default=256, help="HEALPix nside")
    parser.add_argument("--dir_root", type=str, default="/home/zjding/fiberassignment/JUST/BGS_mock/Junyu_mock/fba/output/", help="Root directory of fiber assigned targets")
    parser.add_argument("--fba_filename", type=str, default="fba_tile_{tid}.fits", help="Filename pattern for fiber assigned targets")
    parser.add_argument("--odir", type=str, help="ouput directory of fiber assigned catalog")
    parser.add_argument("--nest", action="store_true", help="Use nested HEALPix ordering")
    args = parser.parse_args()

    Npasses = args.Npasses
    seed = args.seed
    MTL_path = args.MTL_path
    nside = args.nside
    nest = args.nest
    fba_filepath = os.path.join(args.dir_root, f"seed{seed}/")
    fba_filename = args.fba_filename
    odir = args.odir
    
    print(f"nside: {nside}")
    print(f"MTL_path: {MTL_path}")
    print(f"Npasses: {Npasses}")
    print(f"seed: {seed}")
    print(f"mock_version: {args.mock_version}")
    print(f"dir_root: {args.dir_root}")
    print(f"fba_filename: {fba_filename}")
    print(f"nest: {nest}")
    
    all_files = []
    for pid in range(Npasses):
        file_list = sorted(glob.glob(fba_filepath + fba_filename.format(tid=f"{pid+1}*")))
        print(f"pass={pid+1}, {len(file_list)} fba files")
        all_files.extend(file_list)
    
    n_workers = min(16, os.cpu_count() or 1)
    print(f"Processing {len(all_files)} files with {n_workers} workers")
    
    cat_assigned_dict = {}
    tileid_list = []
    
    # fork pool: worker defined above is inherited by child processes on Linux
    ctx = get_context("fork")
    with ctx.Pool(processes=n_workers) as pool:
        for count, (tile_id, cat_assigned) in enumerate(
            pool.imap(process_one_fba_file, all_files, chunksize=32),
            start=1,
        ):
            key = f"TILE_{tile_id}"
            cat_assigned_dict[key] = cat_assigned
            tileid_list.append(tile_id)
            if count % 1000 == 0:
                print(count)
    
    print(f"done: {len(tileid_list)} tiles")

    
    ## one row per TARGETID with TILES and FIBERS hyphen lists
    n_agg_workers = min(16, os.cpu_count() or 1)
    cat_assigned_all = aggregate_assigned_by_target(
        cat_assigned_dict,
        n_workers=n_agg_workers,
    )
    print("N_assigned_all_unique", len(cat_assigned_all))
    
    output = cat_assigned_all
        
    
    ## load the parent catalog and add fiber-assignment completeness weight
    cat_parent = Table.read(MTL_path)

    pix_parent, count_parent = count_targets_per_healpix_pixel(
        cat_parent["ra"], cat_parent["dec"], nside, nest=nest
    )
    pix_output = lookup_healpix_for_targetids(output["TARGETID"], cat_parent["idx"], pix_parent)
    count_fa = np.bincount(pix_output, minlength=hp.nside2npix(nside))
    
    n_parent_pixels = np.count_nonzero(count_parent)
    n_output_pixels = np.count_nonzero(count_fa)
    print(f"Parent targets: {len(cat_parent)}, pixels with n_target>0: {n_parent_pixels}")
    print(f"Assigned output targets: {len(output)}, pixels with n_target>0: {n_output_pixels}")
    
    output = add_completeness_weight(output, cat_parent, nside, nest=nest)
    print(output["COMP_FA"][:10])
    
    
    ## show the heatmap of the completeness weight
    ra_range = (
        float(np.min(cat_parent["ra"])),
        float(np.max(cat_parent["ra"])),
    )
    dec_range = (
        float(np.min(cat_parent["dec"])),
        float(np.max(cat_parent["dec"])),
    )

    if odir == None:
        odir = os.path.join(fba_filepath, "LSScats/")
    os.makedirs(odir, exist_ok=True)

    weight_map = plot_completeness_map(
        count_fa,
        count_parent,
        nside,
        nest=nest,
        n_passes=Npasses,
        ra_range=ra_range,
        dec_range=dec_range,
        outfile=os.path.join(odir, "completeness_cartview.png"),
    )


    weight_map = np.full(hp.nside2npix(nside), hp.UNSEEN, dtype=np.float64)
    assigned_pixels = count_fa > 0
    weight_map[assigned_pixels] = (
        count_fa[assigned_pixels].astype(np.float64) / count_parent[assigned_pixels].astype(np.float64)
    )

    valid_weights = weight_map[assigned_pixels]
    #vmin, vmax = np.percentile(valid_weights, [2, 98])
    print(f"Fiber assignment completeness: min={valid_weights.min():.3f}, max={valid_weights.max():.3f}")
    #print(f"Plot range (2-98 percentile): {vmin:.3f} to {vmax:.3f}")

    fig = plt.figure(figsize=(13, 10))
    hp.cartview(
        weight_map,
        nest=nest,
        coord="C",
        lonra=[-2, 92],
        latra=[-2, 90],
        min=0.0,
        max=1.0,
        title=f"fiber assignment completeness ({Npasses} passes)",
        unit="COMP_FA",
        cmap="RdYlGn",
        flip="astro",
    )
    hp.graticule(dmer=10, dpar=10, verbose=False)
    plt.savefig(os.path.join(odir, "completeness_cartview2.png"), dpi=150)
    print(f"saved cartview plot: {odir}/completeness_cartview2.png")
    plt.close(fig)


    fig = plt.figure(figsize=(13, 10))
    hp.mollview(
        weight_map,
        nest=nest,
        coord="C",
        min=0.0,
        max=1.0,
        title=f"fiber assignment completeness ({Npasses} passes)",
        unit="COMP_FA",
        cmap="RdYlGn",
        flip="astro",
    )
    hp.graticule(dmer=30, dpar=15, verbose=False)
    plt.savefig(os.path.join(odir, "completeness_mollview.png"), dpi=150)
    print(f"saved mollview plot: {odir}/completeness_mollview.png")
    plt.close(fig)

    comb_output = join(output, cat_parent, keys_left=["TARGETID"], keys_right=["idx"])

    ofile = os.path.join(odir, "fba_cat.fits")
    comb_output.write(ofile, overwrite=True)
    print(f"saved catalog: {ofile}")

if __name__ == "__main__":
    main()
