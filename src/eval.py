import json
import numpy as np

def calculate_time_iou(start1, end1, start2, end2):
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start
    iou = intersection / union
    return iou

def r1_at(thred, ious):
    cnt = 0
    for iou in ious:
        if iou >= thred:
            cnt += 1
    return (cnt/len(ious))

def r1_at_dist(thred, dists):
    cnt = 0
    for dist in dists:
        if dist <= thred:
            cnt += 1
    return (cnt/len(dists))

if __name__ == '__main__':

    log_path = ''
    answer_path = ''

    assert log_path != '' and answer_path != ''

    with open(log_path,'r') as file:
        logs = json.load(file)
    with open(answer_path,'r') as f:
        anses = json.load(f)

    ious = []
    dists = []
    in_tokens, out_tokens = 0, 0
    err_cnt = 0
    not_parsed = 0

    for ind in range(len(logs)):
        log = logs[ind]
        res = log['response']
        err = log['error']
        ans = anses[ind]

        if err == '':
            gt_start = ans[0]
            gt_end = ans[1]
            try:
                answer = json.loads(res['content']) # parse response in json
                pred_start = answer['answer'][0]
                pred_end = answer['answer'][1]
                iou = calculate_time_iou(pred_start,pred_end,gt_start,gt_end)
                dist = np.abs(pred_start-gt_start)
                ious.append(iou)
                dists.append(dist)
            except:
                print('The format can not be successfully parsed!')
                ious.append(0)
                dists.append(float('inf'))
                not_parsed += 1
        
        try:
            in_tokens += res['prompt_tokens']
            out_tokens += res['completion_tokens']
        except:
            err_cnt += 1

    
    r1 = r1_at(0.1, ious)
    r3 = r1_at(0.3, ious)
    r5 = r1_at(0.5, ious)
    r7 = r1_at(0.7, ious)
    r9 = r1_at(0.9, ious)
    miou = sum(ious)/len(ious)
    r1s = r1_at_dist(1, dists)
    r3s = r1_at_dist(3, dists)
    r5s = r1_at_dist(5, dists)
    r10s = r1_at_dist(10, dists)
    
    print('--------OUTPUT--------')
    print('Number of samples:\t',len(ious))
    print('---------------------')
    print('r1\t' + round(r1*100,2))
    print('r3\t' + round(r3*100,2))
    print('r5\t' + round(r5*100,2))
    print('r7\t' + round(r7*100,2))
    print('r9\t' + round(r9*100,2))
    print('mIoU\t' + round(miou*100,2))
    print('r1s\t' + round(r1s*100,2))
    print('r3s\t' + round(r3s*100,2))
    print('r5s\t' + round(r5s*100,2))
    print('r10s\t' + round(r10s*100,2))
    print('---------------------')
    print('Number of samples that are not parsed:\t',not_parsed)
    print('Number of errors\t:', err_cnt)