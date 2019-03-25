#!/usr/bin/env python3

import numpy as np
from scipy import signal
from scipy.spatial import cKDTree
from numpy import pi
from scipy.cluster.vq import kmeans
import cv2

try:
    import gputools
    GPUTOOLS = True
except:
    GPUTOOLS = False


def create_correlation_patch(angle_1,angle_2,radius):

    # width and height
    width  = radius*2+1
    height = radius*2+1

    # initialize template
    template = []
    for i in range(4):
        x = np.zeros((height, width))
        template.append(x)

    # midpoint
    mu = radius
    mv = radius

    # compute normals from angles
    n1 = [-np.sin(angle_1), np.cos(angle_1)]
    n2 = [-np.sin(angle_2), np.cos(angle_2)]

    # for all points in template do
    for u in range(width):
        for v in range(height):
            # vector
            vec  = [u-mu, v-mv]
            dist = np.linalg.norm(vec)

            # check on which side of the normals we are
            s1 = np.dot(vec, n1)
            s2 = np.dot(vec, n2)

            if dist <= radius:
                if s1 <= -0.1 and s2 <= -0.1:
                    template[0][v,u] = 1
                elif s1 >= 0.1 and s2 >= 0.1:
                    template[1][v,u] = 1
                elif s1 <= -0.1 and s2 >= 0.1:
                    template[2][v,u] = 1
                elif s1 >= 0.1 and s2 <= -0.1:
                    template[3][v,u] = 1

    # # normalize
    for i in range(4):
        template[i] /= np.sum(template[i])

    return template

def detect_corners_template(gray, template, mode='same'):
    img_corners = [None]*4
    for i in range(4):
        if GPUTOOLS and mode == 'same':
            img_corners[i] = gputools.convolve(gray, template[i])
        else:
            img_corners[i] = signal.convolve(gray, template[i], mode=mode)

    img_corners_mu = np.mean(img_corners, axis=0)

    arr = np.array([img_corners[0]-img_corners_mu, img_corners[1]-img_corners_mu,
                    img_corners_mu-img_corners[2], img_corners_mu-img_corners[3]])
    # case 1: a=white, b=black
    img_corners_1 = np.min(arr, axis=0)

    # case 2: b=white, a=black
    img_corners_2 = np.min(-arr, axis=0)

    # combine both
    img_corners = np.max([img_corners_1, img_corners_2], axis=0)

    return img_corners


TPROPS = [[0, pi/2], [pi/4, -pi/4],
          # [0, pi/4], [0, -pi/4],
          [pi/4, pi/2], [-pi/4, pi/2]]
          # [-3*pi/8, 3*pi/8], [-pi/8, pi/8],
          # [-pi/8, -3*pi/8], [pi/8, 3*pi/8]]
# TPROPS = [[0, pi/2], [0, -pi/4], [0, pi/4]]
RADIUS = [6, 8, 10]

def detect_corners(gray, radiuses=RADIUS):
    out = np.zeros(gray.shape)

    for angle_1, angle_2 in TPROPS:
        for radius in radiuses:
            temp = create_correlation_patch(angle_1, angle_2, radius)
            corr = detect_corners_template(gray, temp)
            out = np.max([corr, out], axis=0)

    return out

def get_corner_candidates(corr, step=40, thres=0.01):
    out = []
    check = set()
    for i in range(0, corr.shape[0], step//2):
        for j in range(0, corr.shape[1], step//2):
            region = corr[i:i+step, j:j+step]
            ix = np.argmax(region)
            r, c = np.unravel_index(ix, region.shape)
            val = region[r, c]
            if val > thres and (r+i, c+j) not in check:
                out.append( (r+i, c+j, val) )
                check.add( (r+i, c+j) )
    return np.array(out)

def non_maximum_suppression(corners, dist=40):
    tree = cKDTree(corners[:, :2])
    good = np.ones(len(corners), dtype='bool')
    for (a, b) in tree.query_pairs(dist):
        if not good[a] or not good[b]:
            continue
        sa = corners[a, 2]
        sb = corners[b, 2]
        if sa >= sb:
            good[b] = False
        else:
            good[a] = False
    return corners[good]

def solve_patch_corner(dx, dy):
    matsum = np.zeros((2,2))
    pointsum = np.zeros(2)
    for i in range(dx.shape[0]):
        for j in range(dx.shape[1]):
            vec = [dy[i,j], dx[i,j]]
            pos = (i,j)
            mat = np.outer(vec, vec)
            pointsum += mat.dot(pos)
            matsum += mat

    try:
        minv = np.linalg.inv(matsum)
    except np.linalg.LinAlgError:
        return None

    newp = minv.dot(pointsum)
    return newp

def get_angle_modes(corners, gray, winsize=11):
    halfwin = (winsize-1)//2
    out = []

    for i, corner in enumerate(corners):
        y, x = corner[:2]
        y = int(round(y))
        x = int(round(x))

        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        gg = gray[y-halfwin:y+halfwin+1, x-halfwin:x+halfwin+1]
        rx = dx[y-halfwin:y+halfwin+1, x-halfwin:x+halfwin+1]
        ry = dy[y-halfwin:y+halfwin+1, x-halfwin:x+halfwin+1]

        angs = np.mod(np.angle(rx + ry*1j).flatten(), np.pi)
        absr = np.abs(rx + ry*1j)
        weights = absr.flatten()
        sim = np.random.choice(angs, p=weights/np.sum(weights), size=len(angs)*5)
        means, distortion = kmeans(sim, 2)

        out.append(means)

    return np.array(out)

def score_corners(corners, gray, winsize=11):
    halfwin = (winsize-1)//2

    scores = np.zeros(corners.shape[0])

    for i, corner in enumerate(corners):
        y, x, score = corner
        y = int(round(y))
        x = int(round(x))

        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        gg = gray[y-halfwin:y+halfwin+1, x-halfwin:x+halfwin+1]
        rx = dx[y-halfwin:y+halfwin+1, x-halfwin:x+halfwin+1]
        ry = dy[y-halfwin:y+halfwin+1, x-halfwin:x+halfwin+1]

        angs = np.mod(np.angle(rx + ry*1j).flatten(), np.pi)
        absr = np.abs(rx + ry*1j)
        weights = absr.flatten()
        sim = np.random.choice(angs, p=weights/np.sum(weights), size=len(angs)*5)
        means, distortion = kmeans(sim, 2)

        patch = create_correlation_patch(means[0], means[1], halfwin)
        new_score = np.max(detect_corners_template(gg, patch, mode='valid'))

        scores[i] = new_score

    return scores


def refine_corners(corners, gray, winsize=11, check_only=False):
    halfwin = (winsize-1)//2

    out = []

    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    for corner in corners:
        y, x, score = corner
        y = int(round(y))
        x = int(round(x))

        rx = dx[y-halfwin:y+halfwin+1, x-halfwin:x+halfwin+1]
        ry = dy[y-halfwin:y+halfwin+1, x-halfwin:x+halfwin+1]

        newp = solve_patch_corner(rx, ry)
        if newp is None:
            continue # bad point

        newp = newp - [halfwin, halfwin]
        if np.any(np.abs(newp) > halfwin+1):
            continue # bad point

        coord = newp + [y, x]
        if check_only:
            out.append([y, x, score])
        else:
            out.append([coord[0], coord[1], score])

    return np.array(out)



def normalize_image(img):
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    blur_size = int(np.sqrt(gray.size) / 2)
    grayb = cv2.GaussianBlur(gray, (3,3), 1)
    gray_mean = cv2.blur(grayb, (blur_size, blur_size))
    diff = (np.float32(grayb)-gray_mean) / 255.0
    diff = np.clip(diff, -0.2, 0.2)+0.2
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    return diff

def checkerboard_score(corners, size=(9,6)):
    corners_reshaped = corners[:, :2].reshape(*size, 2)
    maxm = 0
    for rownum in range(size[0]):
        for colnum in range(1,size[1]-1):
            pts = corners_reshaped[rownum, [colnum-1, colnum, colnum+1]]
            top = np.linalg.norm(pts[2] + pts[0] - 2*pts[1])
            bot = np.linalg.norm(pts[2] - pts[0])
            if np.abs(bot) < 1e-9:
                return 1
            maxm = max(top/bot, maxm)
    for colnum in range(0,size[1]):
        for rownum in range(1, size[0]-1):
            pts = corners_reshaped[[rownum-1, rownum, rownum+1], colnum]
            top = np.linalg.norm(pts[2] + pts[0] - 2*pts[1])
            bot = np.linalg.norm(pts[2] - pts[0])
            if np.abs(bot) < 1e-9:
                return 1
            maxm = max(top/bot, maxm)
    return maxm

def make_mask_line(shape, start, end, thickness=2):
    start = tuple([int(x) for x in start])
    end = tuple([int(x) for x in end])
    mask = np.zeros(shape)
    cv2.line(mask, start, end, 1, thickness)
    return mask


# TODO: this should be replaced by the growing checkerboard from the Geiger et al paper
def reorder_checkerboard(corners, gray, size=(9,6)):
    corners_xy = corners[:, :2]

    tree = cKDTree(corners_xy)
    dist, ix_mid = tree.query(np.median(corners_xy, axis=0))

    corner_mid = corners_xy[ix_mid]
    dists, ixs = tree.query(corner_mid, k=7)

    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    dmag = np.abs(dx + dy*1j)

    ixs = [i for i in ixs[1:] if i < corners_xy.shape[0]]

    mags = []
    for ix in ixs[1:]:
        mask = make_mask_line(
            gray.shape, corner_mid[::-1], corners_xy[ix, ::-1], thickness=3
        )
        mask /= np.sum(mask)
        mag = np.sum(mask * dmag)
        mags.append(mag)

    mags = np.array(mags) / np.max(mags)

    corners_selected = corners_xy[ixs[1:]][mags > 0.7]
    mags = mags[mags > 0.7]

    dirs = corners_selected - corner_mid
    dirs_norm = dirs / np.linalg.norm(dirs, axis=1)[:, None]
    ax1 = dirs[np.argmax(mags)]
    ax2 = dirs[np.argmin(np.abs(np.dot(dirs_norm, ax1)))]

    ax1 *= np.sign(np.sum(ax1))
    ax2 *= np.sign(np.sum(ax2))

    starts = np.argsort(np.dot(corners_xy, ax1 + ax2))

    ixs_best = None
    d_best = np.inf
    # score_best = np.inf
    start_best = 0

    for start in starts[:2]:
        start_xy = corners_xy[start]

        for (ax1_test, ax2_test) in [[ax1,ax2],[ax2,ax1]]:
            ## better estimate of axes
            _, right_ix = tree.query(ax1_test + start_xy)
            _, bot_ix = tree.query(ax2_test + start_xy)
            ax1_new = 0.6*ax1_test + 0.4*(corners_xy[right_ix] - start_xy)
            ax2_new = 0.6*ax2_test + 0.4*(corners_xy[bot_ix] - start_xy)

            xs, ys = np.mgrid[:size[0], :size[1]]
            offsets = xs[:, :, None] * ax1_new + ys[:, :, None] * ax2_new
            points_query = (start_xy + offsets).reshape(-1, 2)
            dists, ixs = tree.query(points_query)
            # score = checkerboard_score(corners[ixs], size)
            d = np.max(dists)
            if d < d_best:
                # score_best = score
                d_best = d
                ixs_best = ixs
                start_best = start


    return np.copy(corners[ixs_best]), d_best


def detect_checkerboard(gray, size=(9,6), winsize=9):
    diff = normalize_image(gray)
    radiuses = [winsize+3]
    if winsize >= 8:
        radiuses.append(winsize-3)

    corr = detect_corners(diff, radiuses=radiuses)

    corrb = cv2.GaussianBlur(corr, (7,7),3)
    corners = get_corner_candidates(corrb, winsize+2, np.max(corrb)*0.2)
    if len(corners) < size[0]*size[1]:
        return None, 1.0

    corners = non_maximum_suppression(corners, winsize-2)
    corners_sp = refine_corners(corners, diff, winsize=winsize+2)
    # corners_sp = refine_corners(corners_sp, diff, winsize=max(winsize//2-1,5),
    #                             check_only=True)
    # corners_sp = refine_corners(corners_sp, diff, winsize=5,
    #                             check_only=True)

    scores = corners_sp[:, 2]

    num_corners = size[0]*size[1]

    best_ix = np.argsort(-scores)[:num_corners+3]
    best_corners = corners_sp[np.sort(best_ix)]
    best_corners, max_dist = reorder_checkerboard(best_corners, diff)

    check_score = checkerboard_score(best_corners, size)

    if len(np.unique(best_corners, axis=0)) < num_corners:
        check_score = 1

    if np.isnan(check_score) or check_score > 0.3:
        # print('trying with extra points...')
        best_ix = np.argsort(-scores)[:num_corners+10]
        best_corners = corners_sp[np.sort(best_ix)]
        best_corners, max_dist = reorder_checkerboard(best_corners, diff)
        check_score = checkerboard_score(best_corners, size)

    # corner_scores = best_corners[:, 2]

    # print('corner_scores', np.mean(corner_scores))
    # print('max dist', max_dist)
    # print('checkerboard score', check_score)

    corners_opencv = np.copy(best_corners[:, :2])
    corners_opencv[:, 0] = best_corners[:, 1]
    corners_opencv[:, 1] = best_corners[:, 0]

    corners_opencv = corners_opencv[:, None]

    if check_score > 0.3 \
       or len(best_corners) < num_corners \
       or max_dist > winsize*3:
        return None, 1.0
    else:
        return corners_opencv, check_score
