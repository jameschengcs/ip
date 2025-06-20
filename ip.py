'''
Image Painter v1.0
Main program with interface 
# (c) 2024, Chang-Chieh Cheng, jameschengcs@nycu.edu.tw
'''

import numpy as np
import torch
import os, sys, random, json
import vgi
from argparse import ArgumentParser
from time import time
from cv2 import VideoWriter, VideoWriter_fourcc
from painter import painter, repaint, accuracyConfig

def main():
    t0 = time()
    if torch.cuda.is_available():
        print('CUDA is available.')

    app_description = "Image Painter v1.0"
    brush_dir = 'brush/'
    brush_extname = '.png'
    def_arg = {
        # config parameters (user control)
        "input_path":"",                    # input filepath
        "output_path":"",                # output image filepath
        "accuracy": 4,                   # accuracy, 0 and 7 indicate the lowest and highest accuracies respectively.
        "brush_name":"watercolor1",      # brush name
        "m":500,                         # maximum stroke number    
        "frame_rate":8,                  # Frame per second (FPS)
        "stroke_batch_size":10,          # painting stroke batch size
        "brush_oppacity": False,         # Using brush template as oppacity
        "output_json": False,            # output json
        "tau_x": None,                     # tau_x, sub-region width
        "tau_y": None,                     # tau_y, sub-region height
        "beta":0.3,                        # beta, difference threshold
        "p": 4,                            # Number of strokes for parallel computing
        "bo": None,                        # Number of rounds of backward optimization
    }    

    parser = ArgumentParser(description=app_description)
    parser.add_argument('input_path', type=str, default=def_arg["input_path"], help='input filepath')
    parser.add_argument('output_path', type=str, default=def_arg["output_path"], help='output filepath')
    parser.add_argument('--a', type=int, default=def_arg["accuracy"], help='accuracy')
    parser.add_argument('--b', type=str, default=def_arg["brush_name"], help='brush name')
    parser.add_argument('--f', type=int, default=def_arg["frame_rate"], help='Frame per second (FPS)')
    parser.add_argument('--m', type=int, default=def_arg["m"], help='maximum stroke number')
    parser.add_argument('--s', type=int, default=def_arg["stroke_batch_size"], help='painting stroke batch size')
    parser.add_argument('--tau_x', type=int, default=def_arg["tau_x"], help='sub-region width')
    parser.add_argument('--tau_y', type=int, default=def_arg["tau_y"], help='sub-region height')
    parser.add_argument('--beta', type=float, default=def_arg["beta"], help='difference threshold')
    parser.add_argument('--p', type=int, default=def_arg["p"], help='Number of strokes for parallel computing')
    parser.add_argument('--bo', type=int, default=def_arg["bo"], help='Number of rounds of backward optimization')
    parser.add_argument('-j', action = 'store_true', default=def_arg["output_json"], help='output json')    
    parser.add_argument('-o', action = 'store_true', default=def_arg["brush_oppacity"], help='using brush template as oppacity')    
    
    args = parser.parse_args()
    

    input_path = args.input_path
    if not os.path.isfile(input_path):
        print('Input file not exist:', args.input_path)
        exit_program()

    brush_name = args.b
    brush_path = brush_dir + brush_name + brush_extname
    if not os.path.isfile(brush_path):
        print('Brush dose not exist: ', brush_path)
        exit_program()

    output_path = args.output_path
    output_dir, output_filename, output_extname = vgi.parsePath(output_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_max = args.m
    accuracy = args.a
    output_json = args.j    
    brush_oppacity = args.o
    brush_masking = not brush_oppacity
    brush_inv = False

    pyd_max_levels = 4
    pyd_min_size = 128
    lv_rep = 1    
    thres_r = args.beta
    div_shape = None
    if args.tau_x is not None:
        if args.tau_y is None:
            args.tau_y = args.tau_x
        div_shape = (args.tau_x, args.tau_y)
    
    thres_r_w = 0.8
    min_thres_r = 0.025
    min_px = 3
    alpha_ini = 0.8    
    min_decline, reopt = accuracyConfig(accuracy)
    if args.bo is not None:
        print('args.bo:', args.bo)
        reopt = args.bo

    search_batch_size = args.p
    print('search_batch_size:', search_batch_size)
    verbose = 1

    input_dir, input_filename, input_extname = vgi.parsePath(input_path)
    if input_extname == '.json':
        with open(input_path) as jFile:
            jData = jFile.read()
            D = json.loads(jData)
            bg = D['bg']
            out_shape = D['shape']
            Q = np.array(D['Q'])
            n_max = Q.shape[0]
            print("Painting with", n_max, "strokes.")
            if output_extname == '.mp4':
                frame_rate = args.f
                stroke_batch_size = np.clip(args.s, 1, n_max)
                frames = repaint(Q, out_shape, bg = bg, batch = stroke_batch_size, save_batch = True, brush_path = brush_path, brush_inv = brush_inv, brush_masking = brush_masking)
                video_size = (out_shape[1], out_shape[0])
                out = VideoWriter(output_path, VideoWriter_fourcc(*'mp4v'), frame_rate, video_size, 1)
                for frame in frames:
                    frame = np.flip(frame * 255, axis = -1).astype('uint8')                     
                    out.write(frame)                    
                out.release()
            else:
                img = repaint(Q, out_shape, bg = bg, brush_path = brush_path, brush_inv = brush_inv, brush_masking = brush_masking)
                vgi.saveImage(output_path, img, revChannel = True)        
    else:
        target = vgi.loadImage(input_path, normalize = True, gray = False )
        print(input_path, target.shape)

        pt = painter(target, pyd_max_levels = pyd_max_levels, pyd_min_size = pyd_min_size, search_batch_size = search_batch_size)
        pt.paint(n_max = n_max, brush_path = brush_path, brush_inv = brush_inv, brush_masking = brush_masking,              
                 reopt = reopt, lv_rep = lv_rep, thres_r = thres_r, thres_r_w = thres_r_w, 
                 min_thres_r = min_thres_r, min_decline = min_decline, div_shape = div_shape, 
                 verbose = verbose)
        if len(output_dir) > 0:
            output_dir += '/'
        json_path = output_dir + output_filename + '.json' if output_json else None
        pt.save(output_path, json_path)
    td = time() - t0
    print('Painting time:', td)  
    sys.exit(0)  

def exit_program():    
    sys.exit(0)

if __name__ == "__main__":
    main()    