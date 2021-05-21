import os
import time
import pickle
import pdb
import glob

import random
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import cv2

import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms

import sys
sys.path.append(os.path.abspath(".."))
from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
# from cirtorch.utils.whiten import whitenlearn, whitenapply, pcawhitenlearn

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
    # new networks with whitening learned end-to-end
    'rSfM120k-tl-resnet50-gem-w'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w'            : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
}

datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']
whitening_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']

class MyClass: pass
args = MyClass()
# args.network_path='retrievalSfM120k-resnet101-gem'; args.whitening='retrieval-SfM-120k'
args.network_path='gl18-tl-resnet152-gem-w'; args.whitening=None
args.multiscale='[1]'    #'[1, 1/2**(1/2), 1/2]'
args.image_size=1024
args.gpu_id="0"
args.data_root='./../data'

def main():
    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network from path
    print(">> Loading network:\n>>>> '{}'".format(args.network_path))
    if args.network_path in PRETRAINED:
        # pretrained networks (downloaded automatically)
        state = load_url(PRETRAINED[args.network_path], model_dir=os.path.join(args.data_root, 'networks'))
    else:
        # fine-tuned network from path
        state = torch.load(args.network_path)

    # parsing net params from meta
    # architecture, pooling, mean, std required
    # the rest has default values, in case that is doesnt exist
    net_params = {}
    net_params['architecture'] = state['meta']['architecture']
    net_params['pooling'] = state['meta']['pooling']
    net_params['local_whitening'] = state['meta'].get('local_whitening', False)
    net_params['regional'] = state['meta'].get('regional', False)
    net_params['whitening'] = state['meta'].get('whitening', False)
    net_params['mean'] = state['meta']['mean']
    #net_params['mean'] = [0.486, 0.457, 0.405]
    net_params['std'] = state['meta']['std']
    #net_params['std'] = [0.234, 0.232, 0.224]
    net_params['pretrained'] = False

    # load network
    net = init_network(net_params)
    net.load_state_dict(state['state_dict'])
    
    # if whitening is precomputed
    if 'Lw' in state['meta']:
        net.meta['Lw'] = state['meta']['Lw']
    
    print(">>>> loaded network: ")
    print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = list(eval(args.multiscale))
    if len(ms)>1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening']:
        msp = net.pool.p.item()
        print(">> Set-up multiscale:")
        print(">>>> ms: {}".format(ms))            
        print(">>>> msp: {}".format(msp))
    else:
        msp = 1
    
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # compute whitening
    if args.whitening is not None:
        start = time.time()

        if 'Lw' in net.meta and args.whitening in net.meta['Lw']:
            
            print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
            
            if len(ms)>1:
                Lw = net.meta['Lw'][args.whitening]['ms']
            else:
                Lw = net.meta['Lw'][args.whitening]['ss']

        else:
            # if we evaluate networks from path we should save/load whitening
            # not to compute it every time
            if args.network_path is not None:
                whiten_fn = args.network_path + '_{}_whiten'.format(args.whitening)
                if len(ms) > 1:
                    whiten_fn += '_ms'
                whiten_fn += '.pth'
            else:
                whiten_fn = None

            if whiten_fn is not None and os.path.isfile(whiten_fn):
                print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
                Lw = torch.load(whiten_fn)

            else:
                raise ValueError('whitening')

    else:
        Lw = None
    # ------------------------------------------------------------------------------------------------------------------

    # evaluate on files
    start = time.time()

    # specify your query and db images
    images = sorted(glob.glob('../../dataset/library/reference_resize_480_640_rename/*.JPG')) # 54
    query = sorted(glob.glob('../../dataset/library/query_resize_rename/*.JPG')) # 42

    bbxs = None

    # extract database and query vectors
    vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)
    query_vecs = extract_vectors(net, query, args.image_size, transform, bbxs=bbxs, ms=ms, msp=msp)

    # convert to numpy
    vecs = vecs.numpy()
    query_vecs = query_vecs.numpy()

    # 転置 
    vecs = vecs.T
    query_vecs = query_vecs.T
    number_of_dataset = vecs.shape[0]
    number_of_query = query_vecs.shape[0]

    # 初期化・パス指定
    topk = 5
    path_folder = "../../dataset/library/annotation"
    
    
    # 近傍探索
    
    # ---------------------------------- 1 direction ----------------------------------
    print("--------------------- 1 direction ---------------------")

    map = np.array([])
    top1 = np.array([])

    start = time.time()
    for i in range(number_of_query):
        dis_list = {}
        print("query: ", query[i][42:])
        for j in range(0, number_of_dataset,4):
            # 近傍距離を比較する．もう既にペアできた画像が探索対象外とする．◎
            # 処理時間を短縮したほう
            dis = np.array([np.linalg.norm(query_vecs[i]-vecs[j+x]) for x in range(4)])  
            distance = np.min(dis)
            idx = np.argmin(dis)

            reference = images[j+idx][54:-4]

            # distance = np.sum([np.min([np.linalg.norm(query_vecs[i+y]-vecs[j+x]) for x in range(4)]) for y in range(2)])
            dis_list[reference] = distance
        
        # print("distance = ", dis_list)
        dis_sort = sorted(dis_list.items(), key=lambda x:x[1])
        #ans_img = np.array([dis_sort[x][0] for x in range(topk)])
        ans_img = np.array([dis_sort[x][0][:-2] for x in range(len(dis_sort))])
        #print("ans_img: ", len(ans_img))
        ans_img = np.array([dis_sort[x][0][:-2] for x in range(topk)])
        print(" ans img: ", ans_img)

        #ap = mAP(path_folder, query[i][42:45], ans_img, topk)
        top = Top1(path_folder, query[i][42:45], ans_img)
        print(" top1 = " + str(top))
        #map = np.append(map, ap)
        top1 = np.append(top1, top)

    print("**************************************************")     
    #print("mAP = ", np.mean(map))
    print("Top-1 = ", np.mean(top1))
    print("time = ", time.time()-start)

    # ---------------------------------- 2 directions ----------------------------------
    print("--------------------- 2 directions ---------------------")

    direction = 2

    # 方向指定
    index_all = np.array([[0,1],[0,2],[0,3],
                        [1,0],[1,2],[1,3],
                        [2,0],[2,1],[2,3],
                        [3,0],[3,1],[3,2]])

    map = np.array([])
    top1 = np.array([])

    start = time.time()
    for i in range(0, number_of_query, 4):
        dis_list = {}
        ran = random.sample(range(4), k=2)
        #print("query: ", query[i+ran[0]][42:], ", ", query[i+ran[1]][42:])
        for j in range(0, number_of_dataset, 4):
            reference = images[j][54:-6]

            '''
            # 近傍距離を比較する．もう既にペアできた画像が探索対象外とする．◎
            # 処理時間を短縮したほう
            index = np.array([0,1,2,3])
            distance = 0
            for d in range(direction):
                dis = np.array([np.linalg.norm(query_vecs[i+ran[d]]-vecs[j+index[x]]) for x in range(4-d)])  
                distance += np.min(dis)
                min_arg = np.argmin(dis)
                index[min_arg], index[3-d] = index[3-d], index[min_arg] # Fisher-Yates
            ''' 
            
            # 方向が重複しないように，24通りすべて計算してから，minを選ぶ．
            keep = np.array([[np.linalg.norm(query_vecs[i+ran[x]]-vecs[j+y]) for x in range(2)] for y in range(4)])
            dis = np.array([np.sum([keep[index_all[y][x]][x] for x in range(2)]) for y in range(12)])
            #dis = np.array([np.sum([np.linalg.norm(query_vecs[i+ran[x]]-vecs[j+index_all[y][x]]) for x in range(2)]) for y in range(12)])
            distance = np.min(dis)  
            

            # distance = np.sum([np.min([np.linalg.norm(query_vecs[i+y]-vecs[j+x]) for x in range(4)]) for y in range(2)])
            dis_list[reference] = distance
        
        # print("distance = ", dis_list)
        dis_sort = sorted(dis_list.items(), key=lambda x:x[1])
        #ans_img = np.array([dis_sort[x][0] for x in range(topk)])
        ans_img = np.array([dis_sort[x][0] for x in range(len(dis_sort))])
        #print(" ans_img: ", ans_img)

        ap = mAP(path_folder, query[i][42:45], ans_img, topk)
        top = Top1(path_folder, query[i][42:45], ans_img)
        #print(" ap = " + str(ap))
        map = np.append(map, ap)
        top1 = np.append(top1, top)

    print("**************************************************")     
    print("mAP = ", np.mean(map))
    print("Top-1 = ", np.mean(top1))
    print("time = ", time.time()-start)

    # ---------------------------------- 3 directions ----------------------------------
    print("--------------------- 3 directions ---------------------")
    direction = 3

    # 方向指定
    index_all = np.array([[0,1,2],[0,1,3],[0,2,1],[0,2,3],[0,3,1],[0,3,2],
                        [1,0,2],[1,0,3],[1,2,0],[1,2,3],[1,3,2],[1,3,0],
                        [2,0,1],[2,0,3],[2,1,0],[2,1,3],[2,3,0],[2,3,1],
                        [3,0,1],[3,0,2],[3,1,0],[3,1,2],[3,2,0],[3,2,1]])

    map = np.array([])
    top1 = np.array([])

    start = time.time()
    for i in range(0, number_of_query, 4):
        dis_list = {}
        ran = random.sample(range(4), k=3)
        #print("query: ", query[i+ran[0]][42:], ", ", query[i+ran[1]][42:], ", ", query[i+ran[2]][42:])
        for j in range(0, number_of_dataset, 4):
            reference = images[j][54:-6]

            '''
            # 近傍距離を比較する．もう既にペアできた画像が探索対象外とする．◎
            # 処理時間を短縮したほう
            index = np.array([0,1,2,3])
            distance = 0
            for d in range(direction):
                dis = np.array([np.linalg.norm(query_vecs[i+ran[d]]-vecs[j+index[x]]) for x in range(4-d)])  
                distance += np.min(dis)
                min_arg = np.argmin(dis)
                index[min_arg], index[3-d] = index[3-d], index[min_arg] # Fisher-Yates
            '''
            
            # 方向が重複しないように，24通りすべて計算してから，minを選ぶ．
            keep = np.array([[np.linalg.norm(query_vecs[i+ran[x]]-vecs[j+y]) for x in range(3)] for y in range(4)])
            dis = np.array([np.sum([keep[index_all[y][x]][x] for x in range(3)]) for y in range(24)])
            #dis = np.array([np.sum([np.linalg.norm(query_vecs[i+ran[x]]-vecs[j+index_all[y][x]]) for x in range(3)]) for y in range(24)])
            distance = np.min(dis)  
                     

            # distance = np.sum([np.min([np.linalg.norm(query_vecs[i+y]-vecs[j+x]) for x in range(4)]) for y in range(3)])
            dis_list[reference] = distance

        # print("distance = ", dis_list)
        dis_sort = sorted(dis_list.items(), key=lambda x:x[1])
        #ans_img = np.array([dis_sort[x][0] for x in range(topk)])
        ans_img = np.array([dis_sort[x][0] for x in range(len(dis_sort))])
        #print(" ans_img: ", ans_img)

        ap = mAP(path_folder, query[i][42:45], ans_img, topk)
        top = Top1(path_folder, query[i][42:45], ans_img)
        #print(" ap = " + str(ap))
        map = np.append(map, ap)
        top1 = np.append(top1, top)

    print("**************************************************")        
    print("mAP = ", np.mean(map))
    print("Top-1 = ", np.mean(top1))
    print("time = ", time.time()-start)
        
    # ---------------------------------- 4 directions ----------------------------------
    print("--------------------- 4 directions ---------------------")
    map = np.array([])
    top1 = np.array([])
    direction = 4

    # 方向指定
    index_all = np.array([[0,1,2,3],[0,1,3,2],[0,2,1,3],[0,2,3,1],[0,3,1,2],[0,3,2,1],
                          [1,0,2,3],[1,0,3,2],[1,2,0,3],[1,2,3,0],[1,3,2,0],[1,3,0,2],
                          [2,0,1,3],[2,0,3,1],[2,1,0,3],[2,1,3,0],[2,3,0,1],[2,3,1,0],
                          [3,0,1,2],[3,0,2,1],[3,1,0,2],[3,1,2,0],[3,2,0,1],[3,2,1,0]])

    start = time.time()
    for i in range(0, number_of_query, 4):
        dis_list = {}
        #print("query: ", query[i][42:45])
        for j in range(0, number_of_dataset, 4):
            reference = images[j][54:-6] 
            '''
            # 近傍距離を比較する．もう既にペアできた画像が探索対象外とする．◎
            min_arg = np.array([])
            distance = 0         
            for d in range(direction):
                dis = np.array([np.where(x in min_arg, np.inf, np.linalg.norm(query_vecs[i+d]-vecs[j+x])) for x in range(4)])
                distance += np.min(dis)
                min_arg = np.append(min_arg, np.argmin(dis))
            '''

            '''
            # 近傍距離を比較する．もう既にペアできた画像が探索対象外とする．◎
            # 処理時間を短縮したほう
            index = np.array([0,1,2,3])
            distance = 0
            for d in range(direction):
                dis = np.array([np.linalg.norm(query_vecs[i+d]-vecs[j+index[x]]) for x in range(4-d)])  
                distance += np.min(dis)
                min_arg = np.argmin(dis)
                index[min_arg], index[3-d] = index[3-d], index[min_arg]
            '''

            
            # 方向が重複しないように，24通りすべて計算してから，minを選ぶ．
            keep = np.array([[np.linalg.norm(query_vecs[i+x]-vecs[j+y]) for x in range(4)] for y in range(4)])
            dis = np.array([np.sum([keep[index_all[y][x]][x] for x in range(4)]) for y in range(24)])
            distance = np.min(dis)
            

            '''
            # 方向が重複しないように，24通りすべて計算してから，類似度のmaxを選ぶ．
            keep = np.array([[np.dot(query_vecs[i+x].T, vecs[j+y]) for x in range(4)] for y in range(4)])
            dis = np.array([np.sum([keep[index_all[y][x]][x] for x in range(4)]) for y in range(24)])
            distance = np.min(-dis)
            '''

            '''
            # 4方向を対応つけるように近傍探索する ×
            index = np.array([0,1,2,3])
            dis_tmp = np.array([])
            for d in range(direction):
                dis = np.array([np.linalg.norm(query_vecs[i+x]-vecs[j+index[x]]) for x in range(4)])
                dis_tmp = np.append(dis_tmp, dis)
                index = np.roll(index, 1)
            distance = np.min(dis_tmp)
            '''

            '''
            # 方向決まりで近傍探索 ×
            distance = np.sum([np.linalg.norm(query_vecs[i+x]-vecs[j+x]) for x in range(4)])
            '''
            
            '''
            # 特徴ベクトルの平均を取ってから，近傍距離を比較する ×
            query_dis = np.mean([query_vecs[i+x] for x in range(4)],axis=0)
            reference_dis = np.mean([vecs[j+x] for x in range(4)], axis=0)
            distance = np.linalg.norm(query_dis - reference_dis)
            dis_list[reference] = distance
            '''
            
            '''
            # 類似度を比較する 〇
            min_arg = np.array([])
            distance = 0
            for d in range(direction):
                dis = np.array([np.where(x in min_arg, 0, np.dot(vecs[j+x].T, query_vecs[i+d])) for x in range(4)])
                #print("dis = ", dis)
                distance += np.min(-dis)
                min_arg = np.append(min_arg, np.argmin(-dis))
            '''
            
            '''
            # 近傍距離を比較する(重複あり) △
            distance = np.sum([np.min([np.linalg.norm(query_vecs[i+y]-vecs[j+x]) for x in range(4)]) for y in range(4)])
            '''

            dis_list[reference] = distance / direction

        # print("distance = ", dis_list)
        dis_sort = sorted(dis_list.items(), key=lambda x:x[1])
        #print("last = ", dis_sort[len(dis_sort)-1][0])
        #ans_img = np.array([dis_sort[x][0] for x in range(topk)])
        ans_img = np.array([dis_sort[x][0] for x in range(len(dis_sort))])
        #print(" ans_img: ", ans_img)

        '''
        # save the result image
        if query[i][39:42] == '005':
            for d in range(4):
                plt.figure(figsize=(16, 8))
                for k in range(5):
                    plt.subplot(1, 5, k+1)
                    plt.title("top{0}:{1}_{2}".format(k+1, ans_img[k], d))
                    img = cv2.imread("../dataset/library/reference_resize_480_640_rename/{0}_{1}.JPG".format(ans_img[k], d))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                plt.savefig("ans_multi_direction_1123/after_{0}_{1}.JPG".format(query[i][39:42], d))
        '''    

        ap = mAP(path_folder, query[i][42:45], ans_img, topk)
        top = Top1(path_folder, query[i][42:45], ans_img)
        #print(" ap = " + str(ap))
        map = np.append(map, ap)
        top1 = np.append(top1, top)

    print("**************************************************")        
    print("mAP = ", np.mean(map))
    print("Top-1 = ", np.mean(top1))
    print("time = ", time.time()-start)

def Top1(path_folder, files_query, ans_img):
    path = "{0}/{1}.txt".format(path_folder, files_query)
    with open(path) as f:
        correct_list = np.array([s.strip() for s in f.readlines()])
    
    if ans_img[0] in correct_list:
        return 1
    else:
        return 0
 

def mAPatK(path_folder, files_query, ans_img, top_k):
    path = "{0}/{1}.txt".format(path_folder, files_query)
    with open(path) as f:
        correct_list = np.array([s.strip() for s in f.readlines()])

    #print(" correct:", correct_list)
    correct_num = len(correct_list)

    correct = 0
    ap = 0

    ans_location = np.array([ans_img[i] for i in range(top_k)])
    
    ap_list = np.array([])
    for x in range(top_k):
        if ans_img[x] in correct_list:
            correct += 1
            ap += correct/(x+1)
    if correct == 0:
        return 0
    else:
        return ap/correct_num

def mAP(path_folder, files_query, ans_img, top_k):
    path = "{0}/{1}.txt".format(path_folder, files_query)
    with open(path) as f:
        correct_list = np.array([s.strip() for s in f.readlines()])

    correct_num = len(correct_list)

    correct = 0
    ap = 0
    
    ap_list = np.array([])
    x = 0
    while correct != correct_num:
        if ans_img[x] in correct_list:
            correct += 1
            ap += correct/(x+1)
        x += 1
        if x == 159:
            break

    if correct == 0:
        return 0
    else:
        return ap/correct_num

    
if __name__ == '__main__':
    main()


