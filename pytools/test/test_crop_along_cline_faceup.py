import nrrd
from centerlines.centerlines import get_center_lines, crop_along_cline_faceup

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("-s", "--src", type=str, help="src image dir")
# parser.add_argument("-d", "--dst", type=str, help="src image dir")
# parser.add_argument("-m", "--mask", type=str, help="src image dir")
# args = parser.parse_args()
#
# src_path = args.src
# dst_path = args.dst
# mask_path = args.mask

src_path = "/data2/home/zhouxiangyong/Data/aorta_seg_data/annotation/scalar_crop/1.2.840.113619.2.55.3.2831164355.372.1510830NmUxwhAKHd2Hbjc1ZU8vhlrUWGcR6.nrrd"
dst_path = "/data2/home/zhouxiangyong/Tmp/1.2.840.113619.2.55.3.2831164355.372.1510830NmUxwhAKHd2Hbjc1ZU8vhlrUWGcR6.nrrd"
mask_path = "/data2/home/zhouxiangyong/Data/aorta_seg_data/annotation/scalar_crop_mask_restore/1.2.840.113619.2.55.3.2831164355.372.1510830NmUxwhAKHd2Hbjc1ZU8vhlrUWGcR6.nrrd"

src_data, _ = nrrd.read(src_path)
src_mask, _ = nrrd.read(mask_path)

cline = get_center_lines(src_mask, min_curve_len=50)
src_data_clwarp, field = crop_along_cline_faceup(src_data, cline)

nrrd.write(dst_path, src_data_clwarp)
