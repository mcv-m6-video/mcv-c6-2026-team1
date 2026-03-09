#!/usr/bin/python3
"""
Evaluate submissions for the AI City Challenge.
"""
import os
import zipfile
import tarfile
import traceback
import numpy as np
import pandas as pd
from PIL import Image
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

# Change: Add TrackEval and json imports
from src.tracking.evaluation.methods import HOTA, Identity
import json

# Change: Custom data dir
AI_CITY_DATA_DIR = "data/AI_CITY_CHALLENGE_2022_TRAIN/train"
def get_gt_data():
    return readData("data/AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_train.txt")
def get_sequence_dir(seq_id):
    return os.path.join(AI_CITY_DATA_DIR, f"S{seq_id:02d}")

# Change: Get camera ROI image path
def get_camera_roi_filepath(cid):
    def get_camera_sequence():
        if 1 <= cid <= 5: return 1
        elif 10 <= cid <= 15: return 3
        elif 16 <= cid <= 40: return 4
        else: raise ValueError(f"{cid:03d} is not in any sequence!")
        
    return f"{AI_CITY_DATA_DIR}/S{get_camera_sequence():02d}/c{cid:03d}/roi.jpg"

def get_args():
    parser = ArgumentParser(add_help=False, usage=usageMsg())
    parser.add_argument("data", help="Path to <predicted_labels>.")
    parser.add_argument('--help', action='help', help='Show this help message and exit')
    parser.add_argument('-m', '--mread', action='store_true', help="Print machine readable results (JSON).")
    parser.add_argument('-c', '--cid', type=int, default=None, help="Camera ID for which to evaluate. If None, Multi-Camera Tracking is performed.")
    return parser.parse_args()


def usageMsg():
    return """  python -m src.eval <ground_truth> <prediction>

Details for expected formats can be found at https://www.aicitychallenge.org/.

See `python -m src.eval --help` for more info.

"""


def getData(fh, fpath, names=None, sep='\s+|\t+|,'):
    """ Get the necessary track data from a file handle.
    
    Params
    ------
    fh : opened handle
        Steam handle to read from.
    fpath : str
        Original path of file reading from.
    names : list<str>
        List of column names for the data.
    sep : str
        Allowed separators regular expression string.
    Returns
    -------
    df : pandas.DataFrame
        Data frame containing the data loaded from the stream with optionally assigned column names.
        No index is set on the data.
    """
    
    try:
        df = pd.read_csv(
            fpath, 
            sep=sep, 
            index_col=None, 
            skipinitialspace=True, 
            header=None,
            names=names,
            engine='python'
        )
        
        return df
    
    except Exception as e:
        raise ValueError("Could not read input from %s. Error: %s" % (fpath, repr(e)))


def readData(fpath):
    """ Read test or pred data for a given track. 
    
    Params
    ------
    fpath : str
        Original path of file reading from.
    Returns
    -------
    df : pandas.DataFrame
        Data frame containing the data loaded from the stream with optionally assigned column names.
        No index is set on the data.
    Exceptions
    ----------
        May raise a ValueError exception if file cannot be opened or read.
    """
    names = ['CameraId','Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld']
        
    if not os.path.isfile(fpath):
        raise ValueError("File %s does not exist." % fpath)
    # Gzip tar archive
    if fpath.lower().endswith("tar.gz") or fpath.lower().endswith("tgz"):
        tar = tarfile.open(fpath, "r:gz")
        members = tar.getmembers()
        if len(members) > 1:
            raise ValueError("File %s contains more than one file. A single file is expected." % fpath)
        if not members:
            raise ValueError("Missing files in archive %s." % fpath)
        fh = tar.extractfile(members[0])
        return getData(fh, tar.getnames()[0], names=names)
    # Zip archive
    elif fpath.lower().endswith(".zip"):
        with zipfile.ZipFile(fpath) as z:
            members = z.namelist()
            if len(members) > 1:
                raise ValueError("File %s contains more than one file. A single file is expected." % fpath)
            if not members:
                raise ValueError("Missing files in archive %s." % fpath)
            with z.open(members[0]) as fh:
                return getData(fh, members[0], names=names)
    # text file
    elif fpath.lower().endswith(".txt"):
        with open(fpath, "r") as fh:
            return getData(fh, fpath, names=names)
    else:
        raise ValueError("Invalid file type %s." % fpath)


# Change: Update print_results to output TrackEval dict
def print_results(summary, mread=False):
    """Print a summary dataframe in a human- or machine-readable format.
    
    Params
    ------
    summary : dict
        Dictionary of evaluation results containing HOTA and IDF1 metrics.
    mread : bool
        Whether to print results in machine-readable format (JSON).
    Returns
    -------
    None
        Prints results to screen.
    """
    if mread:
        print(json.dumps({"results": summary}))
        return
    
    print(f"\n{'Metric':<15} {'Score':<10}")
    print("-" * 30)
    for k, v in summary.items():
        print(f"{k:<15} {v*100:.2f}%" if k in ['HOTA', 'IDF1'] else f"{k:<15} {v}")
    print("-" * 30)
    return


def eval(test, pred, cid=None):
    """ Evaluate submission.

    Params
    ------
    test : pandas.DataFrame
        Labeled data for the test set. Minimum columns that should be present in the 
        data frame include ['CameraId','Id', 'FrameId', 'X', 'Y', 'Width', 'Height'].
    pred : pandas.DataFrame
        Predictions for the same frames as in the test data.
    cid : int
        Optional camera ID for which to filter data. Defaults to None.

    Returns
    -------
    df : Dictionary
        Results from the evaluation
    """
    if test is None:
        return None
    
    # Change: Filter camera data
    if cid is not None:
        test = test[test['CameraId'] == cid]
        pred = pred[pred['CameraId'] == cid]
    
    # Internal evaluation functions
    def removeOutliersROI(df):
        """ Remove outliers from the submitted test df that are outside the region of interest for each camera.
        
        Params
        ------
        df : pandas.dfFrame
            df that should be filtered.
        Returns
        -------
        df : pandas.dfFrame
            Filtered df with only objects within the ROI retained.
        """

        def loadroi(cid):
            """Read the ROI image for a given camera.
        
            Params
            ------
            cid : int
                Camera ID whose ROI image should be retrieved.
            Returns
            -------
            im : numpy.ndarray
                Image stored as a 2-d ndarray.
            """

            # Change: Get ROI from custom path
            imf = get_camera_roi_filepath(cid)
            if not os.path.exists(imf):
                raise ValueError("Missing ROI image for camera %03d." % cid)
            img = Image.open( imf, mode='r').convert('L')
            img.load()
            if img.size[0] > img.size[1]:
                img = img.transpose(Image.TRANSPOSE)
                
            im = np.asarray( img, dtype="uint8" )
            if im.shape[0] > im.shape[1]:
                im = im.T

            return im
        
        def isROIOutlier(row, roi, height, width):
            """Check whether item stored in row is outside the region of interest.
            
            Params
            ------
            row : pandas.Series
                Row of data including, at minimum, the 'X', 'Y', 'Width', and 'Height' columns.
            roi : numpy.ndarray
                ROI image for the camera with the same id as in row['CameraId'].
            height : int
                ROI image height.
            width : int
                ROI image width.
            Returns
            -------
            bool
                Return True if image is an outlier.
            """
            xmin = row['X']
            ymin = row['Y']
            xmax = row['X'] + row['Width']
            ymax = row['Y'] + row['Height']
        
            if xmin >= 0 and xmin < width:
                if ymin >= 0 and ymin < height and roi[ymin, xmin] < 255:
                    return True
                if ymax >= 0 and ymax < height and roi[ymax, xmin] < 255:
                    return True
            if xmax >= 0 and xmax < width:
                if ymin >= 0 and ymin < height and roi[ymin, xmax] < 255:
                    return True
                if ymax >= 0 and ymax < height and roi[ymax, xmax] < 255:
                    return True
            return False

        # Store which rows are not ROI outliers
        df['NotOutlier'] = True

        # Make sure df is sorted appropriately
        df.sort_values(['CameraId', 'FrameId'], inplace=True)
        # Load first ROI image
        tscams = df['CameraId'].unique()
        cid = tscams[0]
        roi = loadroi(cid)
        height, width = roi.shape
        # Loop over objects and check for outliers
        for i, row in df.iterrows():
            if row['CameraId'] != cid:
                cid = row['CameraId']
                roi = loadroi(cid)
                height, width = roi.shape
            if isROIOutlier(row, roi, height, width):
                df.at[i, 'NotOutlier'] = False
                
        return df[df['NotOutlier']].drop(columns=['NotOutlier'])

    def removeOutliersSingleCam(df):
        """Remove outlier objects that appear in a single camera.
        
        Params
        ------
        df : pandas.DataFrame
            Data that should be filtered
        Returns
        -------
        df : pandas.DataFrame
            Filtered data with only objects that appear in 2 or more cameras.
        """
        # get unique CameraId/Id combinations, then count by Id
        cnt = df[['CameraId','Id']].drop_duplicates()[['Id']].groupby(['Id']).size()
        # keep only those Ids with a camera count > 1
        keep = cnt[cnt > 1]
        
        # retrict the data to kept ids
        return df.loc[df['Id'].isin(keep.index)]

    def removeRepetition(df):
        """Remove repetition to ensure that all objects are unique for every frame.

        Params
        ------
        df : pandas.DataFrame
            Data that should be filtered
        Returns
        -------
        df : pandas.DataFrame
            Filtered data that all objects are unique for every frame.
        """

        df = df.drop_duplicates(subset=['CameraId', 'Id', 'FrameId'], keep='first')

        return df
        
    # Change: Replace compare_dataframes_mtmc and motmetrics logic with TrackEval ---
    def compare_dataframes_trackeval(gts, ts):
        def iou(a, b):
            if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)), dtype=np.float32)
            inter = np.maximum(0.0, np.minimum(a[:, 2:3], b[:, 2]) - np.maximum(a[:, 0:1], b[:, 0])) * \
                    np.maximum(0.0, np.minimum(a[:, 3:4], b[:, 3]) - np.maximum(a[:, 1:2], b[:, 1]))
            return (inter / (((a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]))[:, None] + ((b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])) - inter + 1e-12)).astype(np.float32)

        gt_map = {id_val: i for i, id_val in enumerate(gts['Id'].unique())}
        tr_map = {id_val: i for i, id_val in enumerate(ts['Id'].unique())}
        data = {"gt_ids": [], "tracker_ids": [], "similarity_scores": [], "num_gt_ids": len(gt_map), "num_tracker_ids": len(tr_map), "num_gt_dets": 0, "num_tracker_dets": 0}

        gt_grp, ts_grp = gts.groupby(['CameraId', 'FrameId']), ts.groupby(['CameraId', 'FrameId'])
        all_frames = pd.concat([gts[['CameraId', 'FrameId']], ts[['CameraId', 'FrameId']]]).drop_duplicates().sort_values(by=['CameraId', 'FrameId']).values
        
        for cid, fid in all_frames:
            g_f = gt_grp.get_group((cid, fid)) if (cid, fid) in gt_grp.groups else pd.DataFrame(columns=['Id', 'X', 'Y', 'Width', 'Height'])
            t_f = ts_grp.get_group((cid, fid)) if (cid, fid) in ts_grp.groups else pd.DataFrame(columns=['Id', 'X', 'Y', 'Width', 'Height'])

            # Get bboxes in xyxy format
            g_box, t_box = g_f[['X', 'Y', 'Width', 'Height']].values.astype(np.float32), t_f[['X', 'Y', 'Width', 'Height']].values.astype(np.float32)
            if len(g_box): g_box[:, 2:] += g_box[:, :2]
            if len(t_box): t_box[:, 2:] += t_box[:, :2]

            # 0-indexed IDs for TrackEval
            g_ids, t_ids = np.array([gt_map[i] for i in g_f['Id']], dtype=np.int64), np.array([tr_map[i] for i in t_f['Id']], dtype=np.int64)
            data["gt_ids"].append(g_ids); data["tracker_ids"].append(t_ids); data["similarity_scores"].append(iou(g_box, t_box))
            data["num_gt_dets"] += len(g_ids); data["num_tracker_dets"] += len(t_ids)

        h, i = HOTA().eval_sequence(data), Identity().eval_sequence(data)
        return {"HOTA": float(np.mean(h["HOTA"])), "IDF1": float(i["IDF1"])}

    # filter ROI outliers
    test = removeOutliersROI(test)
    pred = removeOutliersROI(pred)

    # Change: Filter tracks that appear in only one camera (only for Multi-Camera Tracking)
    if cid is None:
        pred = removeOutliersSingleCam(pred)

    pred = removeRepetition(pred)
    
    # evaluate results
    return compare_dataframes_trackeval(test, pred)


def usage(msg=None):
    """ Print usage information, including an optional message, and exit. """
    if msg:
        print("%s\n" % msg)
    print("\nUsage: %s" % usageMsg())
    exit()


if __name__ == '__main__':
    args = get_args()
    
    test = get_gt_data()
    pred = readData(args.data)
    try:
        summary = eval(test, pred, args.cid)
        print_results(summary, mread=args.mread)
    except Exception as e:
        if args.mread:
            print('{"error": "%s"}' % repr(e))
        else: 
            print("Error: %s" % repr(e))
        traceback.print_exc()
