# This code utilizes the dash_player component which was released under the MIT license

import base64
from datetime import datetime as dt, timedelta
import io
import os
from os.path import exists
from zipfile import ZipFile
import zipfile

import dash_player

import math

import dash
from dash import no_update, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
from dash_extensions import Download # utilized due to Download being bugged in current dash release (2.0)
from dash_extensions.snippets import send_file

import cv2
import numpy as np
import pandas as pd
from flask import Flask, Response
import glob

import time

external_stylesheets = [dbc.themes.SANDSTONE]

server = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True, server=server)

path = 'data/'

# Used for creating full experiment video
class ProcessVideo(object):
    def __init__(self, video, path):
        self.video = video
        self.path = path

    def __del__(self):
        self.video.release()

    # Creates a full experiment video
    def get_video(self):
        eyef = pd.read_csv('data/' + self.path + '/eye_track_2D_hit.tsv', delimiter= '\t')
        vf = pd.read_csv('data/' + self.path + '/video_events.tsv', delimiter= '\t')
        
        f_width = int(self.video.get(3))
        f_height = int(self.video.get(4))
        fps = int(self.video.get((cv2.CAP_PROP_FPS)))
        print(f_width, " vid width")
        print(f_height, " vid height")

        #scale = 80 # percent of original size
        #f_width = int(f_width * scale / 100)
        #f_height = int(f_height * scale / 100)

        x = list(eyef['x'])
        y = list(eyef['y'])
        board = list(eyef['board'])
        dfglobal = list(eyef['global time'])
        dfglobal = pd.to_datetime(dfglobal, format='%M:%S.%f')

        vidfglobal = list(vf['global'])
        vidfglobal = pd.to_datetime(vidfglobal, format='%M:%S.%f')
        vid_time = list(vf['vid time (sec)'])
        video_action = list(vf['action'])

        timer = dfglobal[0]
        print(len(dfglobal) - 1, " this is dfglobal - 1 length")
        print(dfglobal[0], " dfglobal 0")
        #print(dfglobal[11473], " dfglobal 11473")
        #print (dfglobal[11474], " dfglobal 11474")

        output = cv2.VideoWriter('assets/' + self.path + '_full_experiment.mp4', cv2.VideoWriter_fourcc(*'mpv4'),fps,(f_width,f_height))#*'mp4v'
        vid_index = 1 #set url is initial one read
        count = 0
        #print(vid_time[1], " video time at 1")
        time_repeat = 0
        index = 0
        onVideo = False
        readFrame = False
        pause = False
        loop = 0
        count = 0
        nomatch = False
        alpha = 0.5
        
        duplicate = 0
        for i in dfglobal: # deals with multiple values at one time portion at experiment bootup
            if (i == dfglobal[0]):
                duplicate = duplicate + 1
            else:
                break

        while duplicate > 3:
            index = index + 1
            duplicate = duplicate - 1

        start = index

        while timer <= dfglobal[len(dfglobal) - 1]:
            duplicate = 0
            for p in range(index, len(dfglobal)):
                if (dfglobal[p] == dfglobal[index]):
                    duplicate = duplicate + 1
                else:
                    break
            if (duplicate > 3):
                while duplicate > 3:
                    index = index + 1
                    duplicate = duplicate - 1
            #print(timer, " this is timer")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if vid_index >= len(vidfglobal):
                print(timer, " this is timer after vid is greater or equal to last entry")
            print(dfglobal[index], " this is dfglobal (start of loop)")
            print(timer, " this is timer (start of loop)")
            if timer == dfglobal[index]:
                for i in range(3): # determine how many times the global time in eye_track_2D_hit is repeated in the 100 ms value
                    if index + i >= len(dfglobal):
                        break
                    if (dfglobal[index] == dfglobal[index + i]):
                        time_repeat += 1
                    else:
                        break
            if time_repeat == 1:
                loop = -1
                nomatch = False
            elif timer != dfglobal[index]:
                loop = -1
                nomatch = True
            elif time_repeat == 2:
                loop = 1
                nomatch = False
            elif time_repeat == 3:
                loop = 2
                nomatch = False
            while (count < 3):
                if (board[index] == "whiteboard (video)"):
                    if (y[index] >= .616 and y[index] <= .95):
                        onVideo = True
                    else:
                        onVideo = False
                if (onVideo):
                    xval = (((x[index] - 0) * f_width) / 1) + 0
                    yval = (((y[index] - .616) * f_height) / .334) + 0
                    yval = f_height - yval # to account for different coordinate grid

                if (index == start): # first run through get first video frame
                    check, frame = self.video.read()
                    readFrame = False
                elif (vidfglobal[0] > dfglobal[index]):
                    readFrame = False
                else:
                    readFrame = True
                if vid_index < len(vidfglobal): # account for end of vid rows
                    #print(vid_index, " this is vid_index")
                    print(vidfglobal[vid_index], " vid global at vid index")
                    #print(dfglobal[index], " df global at index")
                    if vidfglobal[vid_index] <= dfglobal[index]:
                        #print(video_action[vid_index], " video action")
                        #print(math.isnan(vid_time[vid_index]), " nan check")
                        if video_action[vid_index] == 'set_time' and not math.isnan(vid_time[vid_index]): #handle video actions
                            #print("check 1")
                            if (video_action[vid_index + 1] == 'pause' and vid_time[vid_index] != vid_time[vid_index + 1]):
                                #print(vid_time[vid_index], " vid_time at vid_index")
                                #print(timer, " timer for set time")
                                #print(vid_index, " this is vid_index")
                                #print(self.video.get(cv2.CAP_PROP_POS_MSEC), " current video time in ms")
                                self.video.set(cv2.CAP_PROP_POS_MSEC,vid_time[vid_index] * 1000)
                                pause = True
                                vid_index = vid_index + 1
                            elif (video_action[vid_index + 1] == 'pause'):
                                pause = True
                                vid_index = vid_index + 1
                            #print("check 2")
                        elif (video_action[vid_index] == 'play' and math.isnan(vid_time[vid_index])):
                            print("this is the nan trigger for play")
                            vid_time[vid_index] = 0
                            pause = False
                        elif video_action[vid_index] == 'set_time':
                            vid_index = vid_index + 1
                        elif (video_action[vid_index] == 'play'):
                            pause = False
                        vid_index = vid_index + 1

                if (not pause and readFrame):
                    print("this is triggered")
                    check, frame = self.video.read()
                #print(vid_index, " vid_index in main loop")
                if check == True:
                    if onVideo:
                        #copy_frame = frame.copy()
                        display = cv2.circle(frame, (int(xval),int(yval)), 20, (255,0,0), -1)
                        #display = cv2.addWeighted(display, alpha, frame, 1 - alpha, 0)
                        output.write(display)
                        cv2.imshow('Video', display)
                    else:
                        output.write(frame)
                        cv2.imshow('Video', frame)
                else:
                    break
                loop = loop - 1
                if (loop >= 0): # if repeated eye track hit for global time
                    index = index + 1
                count = count + 1
            count = 0
            if (not nomatch):
                #print("not nomatch increase index")
                index = index + 1
            timer = timer + timedelta(milliseconds=100)
            time_repeat = 0
        print(timer, " end time for timer")

# updates the slider to show the normal video time with the processed video
def update_slider(input, file):
    #print(input, " slider input")
    if (input is not None):
        eyef = pd.read_csv('data/' + file + '/eye_track_2D_hit.tsv', delimiter= '\t')
        vf = pd.read_csv('data/' + file + '/video_events.tsv', delimiter= '\t')
        dfglobal = list(eyef['global time'])
        dfglobal = pd.to_datetime(dfglobal, format='%M:%S.%f')

        vidfglobal = list(vf['global'])
        vidfglobal = pd.to_datetime(vidfglobal, format='%M:%S.%f')
        vid_time = list(vf['vid time (sec)'])
        video_action = list(vf['action'])

        timer = dfglobal[0]
        timer = timer + timedelta(seconds = input)
        #print(timer, " this is timer")
        index = 0
        entry = False
        for i in vidfglobal:
            #print(timer, " this is timer")
            #print(i, " this is i")
            if (i == timer):
                entry = True
                break
            elif (i > timer):
                break
            index = index + 1

        if not entry:
            if index > 0:
                index = index - 1

        if video_action[index] == 'set_time' and not math.isnan(vid_time[index]): #handle video actions
            if (video_action[index + 1] == 'pause' and vid_time[index] != vid_time[index + 1]):
                #print(vid_time[index], " vid_time at vid_index")
                #print(timer, " timer for set time")
                #print(vid_index, " this is vid_index")
                #print(self.video.get(cv2.CAP_PROP_POS_MSEC), " current video time in ms")
                return int(vid_time[index])
                #paused
            elif (video_action[index + 1] == 'pause'):
                #print("pause in first elif is triggered")
                return int(vid_time[index])
        elif video_action[index] == 'pause':
            #print("this basic pause is triggered")
            return int(vid_time[index])
        elif (video_action[index] == 'play' and math.isnan(vid_time[index])):
            #print("this basic play is triggered")
            sec = timer - vidfglobal[index]
            sec = sec.total_seconds()
            #print(sec, " play calc. result")
            return int(sec)
        elif (video_action[index] == 'play'):
            #print("this basic play is triggered")
            sec = timer - vidfglobal[index]
            sec = sec.total_seconds()
            #print(sec, " play calc. result")
            return int(vid_time[index] + sec)
        elif (video_action[index] == 'set_url'):
            #print("set url is triggered")
            return int(vid_time[index])
    else:
        #print("return else is triggered")
        return 0


def check_play(input, file):
    print(input, " check play input")
    if (input is not None):
        eyef = pd.read_csv('data/' + file + '/eye_track_2D_hit.tsv', delimiter= '\t')
        vf = pd.read_csv('data/' + file + '/video_events.tsv', delimiter= '\t')
        dfglobal = list(eyef['global time'])
        dfglobal = pd.to_datetime(dfglobal, format='%M:%S.%f')

        vidfglobal = list(vf['global'])
        vidfglobal = pd.to_datetime(vidfglobal, format='%M:%S.%f')
        vid_time = list(vf['vid time (sec)'])
        video_action = list(vf['action'])

        timer = dfglobal[0]
        timer = timer + timedelta(seconds = input)
        #print(timer, " this is timer")
        index = 0
        entry = False
        for i in vidfglobal:
            #print(timer, " this is timer")
            #print(i, " this is i")
            if (i == timer):
                entry = True
                break
            elif (i > timer):
                break
            index = index + 1

        if not entry:
            if index > 0:
                index = index - 1

        if (video_action[index] == 'play'):
            print("play is true")
            return True
        else:
            return False
    else:
        #print("return else is triggered")
        return False


def check_set_time(input, file):
    #print(input, " check set time input")
    if (input is not None):
        eyef = pd.read_csv('data/' + file + '/eye_track_2D_hit.tsv', delimiter= '\t')
        vf = pd.read_csv('data/' + file + '/video_events.tsv', delimiter= '\t')
        dfglobal = list(eyef['global time'])
        dfglobal = pd.to_datetime(dfglobal, format='%M:%S.%f')

        vidfglobal = list(vf['global'])
        vidfglobal = pd.to_datetime(vidfglobal, format='%M:%S.%f')
        vid_time = list(vf['vid time (sec)'])
        video_action = list(vf['action'])

        timer = dfglobal[0]
        timer = timer + timedelta(seconds = input)
        #print(timer, " this is timer")
        index = 0
        entry = False
        for i in vidfglobal:
            #print(timer, " this is timer")
            #print(i, " this is i")
            if (i == timer):
                entry = True
                break
            elif (i > timer):
                break
            index = index + 1

        if not entry:
            if index > 0:
                index = index - 1
        
                
        if video_action[index] == 'set_time' and vid_time[vid_index] != 'nan': #handle video actions
            if (video_action[index + 1] == 'pause' and vid_time[index] != vid_time[index + 1]):
                #print(vid_time[index], " vid_time at vid_index")
                #print(timer, " timer for set time")
                #print(vid_index, " this is vid_index")
                #print(self.video.get(cv2.CAP_PROP_POS_MSEC), " current video time in ms")
                return vid_time[index]
                #paused
            elif (video_action[index + 1] == 'pause'):
                #print("pause in first elif is triggered")
                return vid_time[index]
        elif (video_action[index] == 'set_url'):
            #print("set url is triggered")
            return vid_time[index]
        else:
           return -1
    else:
        #print("return else is triggered")
        return -1

# used to find index of third _ in imgs
def findIndex(string, char, num):
    count = 0
    for i in range(len(string)):
        if string[i] == char:
            count = count + 1

        if count == num:
            return i
    return -1

# currently not utilized, creates a video from the whiteboard images
def create_video(path):
    whiteboard_imgs = []
    whiteboard_names = []
    name_list = []
    for filename in glob.glob('data/' + path + '/**/*.png'):
        #print(filename, " this is filename")
        image = cv2.imread(filename)
        height = image.shape[0]
        width = image.shape[1]
        whiteboard_imgs.append(image)
        whiteboard_names.append(filename)

    for filepath in whiteboard_names:
        start = filepath.find('Save_')
        end = filepath.find('.png')
        name = filepath[start+5:end]
        print(name)
        name_list.append(name)

    img_output = cv2.VideoWriter('assets/' + path + '_whiteboard.mp4', cv2.VideoWriter_fourcc(*'mp4v'),1,(width,height))
    for i in range(len(whiteboard_imgs)):
        img_output.write(whiteboard_imgs[i])
    img_output.release()


# creates the gaze analysis visualization
def create_gaze(start, duration, files):
    number_of_files = 0
    dataf = []
    videof = []
    final_array = [] # store the frame arrays for each file
    for file in files:
        dataf.append(pd.read_csv('data/' + file + '/eye_track_2D_hit.tsv', delimiter= '\t'))
        videof.append(pd.read_csv('data/' + file + '/video_events.tsv', delimiter= '\t'))
        number_of_files = number_of_files + 1
    print("create gaze triggered")
    print(start, " start")
    print(duration, " duration")
    print(number_of_files, " this is number of files")
    #print(board, " board")
    video = cv2.VideoCapture('https://vel.engr.uga.edu/apps/VRDEO/raydiagrams.mp4')
    f_width = int(video.get(3))
    f_height = int(video.get(4))
    fps = int(video.get((cv2.CAP_PROP_FPS)))
    output = cv2.VideoWriter('assets/gaze.mp4', cv2.VideoWriter_fourcc(*'mpv4'),fps,(f_width,f_height))#*'mp4v'

    colon = start.find(':')
    minute = int(start[0:colon])
    seconds = int(start[colon + 1:len(start)])
    seconds = float(seconds + minute * 60)
    #second_dateTime = pd.to_datetime(seconds, format='%S')
    #duration_dateTime = pd.to_datetime(duration, format='%S')

    i = 0

    frame_array = []
    while (i < number_of_files):
        video.set(cv2.CAP_PROP_POS_MSEC,seconds * 1000)
        frame_array = []
        print(i, " this is i iteration in number of files")
        #video.set(cv2.CAP_PROP_POS_MSEC,seconds * 1000)
        x = list(dataf[i]['x'])
        y = list(dataf[i]['y'])
        eye_board = list(dataf[i]['board'])
        dfglobal = list(dataf[i]['global time'])
        dfglobal = pd.to_datetime(dfglobal, format='%M:%S.%f')

        vidglobal = list(videof[i]['global'])
        vidglobal = pd.to_datetime(vidglobal, format='%M:%S.%f')
        vid_time = list(videof[i]['vid time (sec)'])
        video_action = list(videof[i]['action'])

        in_video = []
        increment = 0
        for j in vid_time:
            #print(j, " this is j")
            if not math.isnan(j):
                if ((seconds <= j) and (seconds + duration >= j)):
                    in_video.append(increment)
                    #print(j, " this is j")
                elif (seconds > j and video_action[increment] == 'play' and increment + 1 < len(vid_time) and vid_time[increment + 1] > seconds):
                    in_video.append(increment)
                    print(j, " this is j in second if")
            else:
                #print("else section")
                if (video_action[increment] == 'play' and increment + 1 < len(vid_time) and vid_time[increment + 1] > seconds):
                    in_video.append(increment)
                    #print("special nan case")
            increment = increment + 1 

        print(len(in_video), " this is the number in video")
        start_time = []
        end_time = []
        current = []
        current_end = []
        start_vid_index = []
       
        for index in in_video:
            #print(index, " this is index")
            if math.isnan(vid_time[index]):
                vid_time[index] = 0
                print(vid_time[index], " vid index after added if was hit")
            if index - 1 not in in_video:
                if video_action[index] == 'set_time': #handle video actions
                    if (video_action[index + 1] == 'pause' and vid_time[index] == vid_time[index + 1]):
                        temp_vid = vid_time[index] - seconds
                        temp_start = vidglobal[index] - timedelta(seconds=temp_vid)
                        start_time.append(temp_start)
                        current.append(seconds)
                        start_vid_index.append(index)
                    #paused
                elif index + 1 not in in_video and video_action[index] == 'play':
                    print("the special play triggered")
                    temp_vid = seconds - vid_time[index]
                    temp_start = vidglobal[index] + timedelta(seconds=temp_vid)
                    start_time.append(temp_start)
                    current.append(seconds)
                    start_vid_index.append(index)

                    temp_vid = seconds + duration - vid_time[index]
                    temp_end = vidglobal[index] + timedelta(seconds=temp_vid)
                    end_time.append(temp_end)
                    current_end.append(seconds + duration)
                    print(temp_end, " this is temp end in play thing")
                elif video_action[index] == 'play':
                    temp_vid = seconds - vid_time[index]
                    temp_start = vidglobal[index] + timedelta(seconds=temp_vid)
                    start_time.append(temp_start)
                    current.append(seconds)
                    start_vid_index.append(index)
                elif video_action[index] == 'set_url':
                    start_time.append(vidglobal[index])
                    current.append(vid_time[index])
                    start_vid_index.append(index)
            elif (video_action[index] == 'set_time' and vid_time[index] != 'nan'):
                if (index + 1 in in_video and video_action[index + 1] == 'pause' and vid_time[index] != vid_time[index + 1]):
                    end_time.append(vidglobal[index + 1])
                    current_end.append(vid_time[index + 1])

                    start_time.append(vidglobal[index])
                    current.append(vid_time[index])
                    start_vid_index.append(index)
                    #print("end and start time append in change time ----------------------------------")
                elif (video_action[index + 1] == 'pause' and vid_time[index] != vid_time[index + 1]):
                    start_time.append(vidglobal[index])
                    current.append(vid_time[index])
                    start_vid_index.append(index)
            elif video_action[index] == 'set_url':
                start_time.append(vidglobal[index])
                current.append(vid_time[index])
                start_vid_index.append(index)
            elif (video_action[index] == 'play' and index + 1 not in in_video):
                temp_vid = seconds + duration - vid_time[index]
                temp_end = vidglobal[index] + timedelta(seconds=temp_vid)
                end_time.append(temp_end)
                current_end.append(seconds + duration)
            elif (video_action[index] == 'pause' and index + 1 >= len(vidglobal)):
                end_time.append(dfglobal[len(dfglobal) - 1])
                current_end.append(vid_time[index])
    
        print(len(start_time), " start time length")
        print(len(end_time), " end time length")
        cycles = []
        timer_array = []

        count = 0
        time_repeat = 0
        onVideo = False
        readFrame = False
        loop = 0
        count = 0
        nomatch = False
        alpha = 0.5
        time_repeat = []
        loop_array = []
        nomatch_array = []
        onVideo_array = []
        isPaused = []

        # get the cycle for each time range
        for l in start_time:
            timer = l
            time_parse = timer.strftime('%M:%S.%f')
            time_parse = time_parse[:-5]
            timer = pd.to_datetime(time_parse, format='%M:%S.%f')
            timer_array.append(timer)
            cycle = find_time(timer, dfglobal)
            cycles.append(cycle)
            time_repeat.append(0)
            loop_array.append(0)
            nomatch_array.append(False)
            onVideo_array.append(False)
            isPaused.append(False)

        iteration = 0 #checks what iteration of start and end times we are on
        current_indices = []

        current_vid_time = seconds
        final = seconds + duration
        print(current_vid_time, " this is the start time")
        print(final, " this is the final time")

        #check, frame = video.read()
        #rectangle = cv2.selectROI(frame, fromCenter=False)
        check, frame = video.read()
        readFrame = False
        begin = True
        #inc = 1
        xArray = []
        yArray = []
        while current_vid_time <= final:
            #print(current_vid_time, " this is current vid time")
            current_indices.clear()
            for h in range(len(current)):
                if (current[h] <= current_vid_time and current_end[h] >= current_vid_time):
                    current_indices.append(h)

            #print(cycle, " this is cycle")
            #print(timer, " timer")
            for h in current_indices:
                if timer_array[h] == dfglobal[cycles[h]]:
                    for k in range(3): # determine how many times the global time in eye_track_2D_hit is repeated in the 100 ms value
                        if cycles[h] + k >= len(dfglobal):
                            break
                        if (dfglobal[cycles[h]] == dfglobal[cycles[h] + k]):
                            time_repeat[h] += 1
                        else:
                            break
            #print(time_repeat, " this is time repeat!!!!!!!!!!!!!!!!")
                if time_repeat[h] == 1:
                    loop_array[h] = -1
                    nomatch_array[h] = False
                elif timer_array[h] != dfglobal[cycles[h]]:
                    loop_array[h] = -1
                    nomatch_array[h] = True
                elif time_repeat[h] == 2:
                    loop_array[h] = 1
                    nomatch_array[h] = False
                elif time_repeat[h] == 3:
                    loop_array[h] = 2
                    nomatch_array[h] = False

                if (vidglobal[start_vid_index[h]] <= dfglobal[cycles[h]]):
                    if video_action[start_vid_index[h]] == 'set_time': #handle video actions
                        if (video_action[start_vid_index[h] + 1] == 'pause'):
                            isPaused[h] = True
                        start_vid_index[h] = start_vid_index[h] + 1 # set time and pause always at same time so always increment after set time
                    elif video_action[start_vid_index[h]] == 'play':
                        isPaused[h] = False
                    start_vid_index[h] = start_vid_index[h] + 1
            while (count < 3):
                #print("count trigger ", count)
                if (begin):
                    readFrame = False
                    begin = False
                elif (True in isPaused):
                    readFrame = False
                    #print(readFrame, " this is readframe")
                    #print(readFrame, " this is readframe")
                else:
                    readFrame = True
                    #print(readFrame, " this is readframe")

                #print(readFrame, " this is readFrame")
                for h in current_indices:
                    if (eye_board[cycles[h]] == "whiteboard (video)"):
                        if (y[cycles[h]] >= .616 and y[cycles[h]] <= .95):
                            onVideo_array[h] = True
                        else:
                           onVideo_array[h] = False
                    #print(onVideo, " this is onvideo")
                        
                    if readFrame:
                        #print (count, " this is count in readFrame")
                        #print("readFrame is true")
                        if (onVideo_array[h]):
                            xval = (((x[cycles[h]] - 0) * f_width) / 1) + 0
                            yval = (((y[cycles[h]] - .616) * f_height) / .334) + 0
                            yval = f_height - yval # to account for different coordinate grid
                            xArray.append(xval)
                            yArray.append(yval)
                    else:
                        if isPaused[h]:
                           if (onVideo_array[h]):
                                xval = (((x[cycles[h]] - 0) * f_width) / 1) + 0
                                yval = (((y[cycles[h]] - .616) * f_height) / .334) + 0
                                yval = f_height - yval # to account for different coordinate grid
                                xArray.append(xval)
                                yArray.append(yval)
                        elif True not in isPaused:
                            if (onVideo_array[h]):
                                xval = (((x[cycles[h]] - 0) * f_width) / 1) + 0
                                yval = (((y[cycles[h]] - .616) * f_height) / .334) + 0
                                yval = f_height - yval # to account for different coordinate grid
                                xArray.append(xval)
                                yArray.append(yval)

                    loop_array[h] = loop_array[h] - 1
                    if (loop_array[h] >= 0): # if repeated eye track hit for global time
                        cycles[h] = cycles[h] + 1
                if readFrame:
                    for n in range(len(xArray)):
                        frame = cv2.circle(frame, (int(xArray[n]),int(yArray[n])), 20, (0,0,255), -1)
                    frame_array.append(frame)
                    check, frame = video.read()
                count = count + 1
            count = 0
                    #print(timer, " this is timer")
                    #print(dfglobal[cycle], " this is dfglobal")
            for h in current_indices:
                if (not nomatch_array[h]):
                    cycles[h] = cycles[h] + 1
                timer_array[h] = timer_array[h] + timedelta(milliseconds=100)
                time_repeat[h] = 0
            if (True not in isPaused):
                current_vid_time = current_vid_time + .100
            if current_vid_time > final:
                for n in range(len(xArray)):
                    frame = cv2.circle(frame, (int(xArray[n]),int(yArray[n])), 20, (0,0,255), -1)
                frame_array.append(frame)
        final_array.append(frame_array)
        i = i + 1 # cycle through selected experiments
    print("before write to video")
    #combine_array = []
    for j in range(len(final_array[0])):
        #print(i, " this is i for range len final_array 0")
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        #print(j, " this is j")
        for i in range(len(final_array)):
            #print(i, " this is i")
            if i + 1 < len(final_array):
                frame = final_array[i][j]
                next = final_array[i + 1][j]
                fin = cv2.addWeighted(next, 0.5, frame, 0.5, 0)
                final_array[i + 1][j] = fin
                #output.write(frame)
                #cv2.imshow('Heatmap', frame)
            else:
                if (j + 1 < len(final_array[0])):
                    frame = final_array[i][j]
                    next = final_array[i][j + 1]
                    fin = cv2.addWeighted(next, 0.5, frame, 0.5, 0)
                    final_array[i][j + 1] = fin
                #output.write(final_array[i][j])
                #cv2.imshow('Heatmap', final_array[i][j])
        output.write(final_array[i][j])
        cv2.imshow('Gaze', final_array[i][j])

    video.release()
    # closes video frames
    cv2.destroyAllWindows()
    print("Create Gaze End")
        #inROI = False
    return html.Div([
        html.Video(
            controls = True,
            id = 'gaze_player',
            #src = app.get_asset_url(filename + '_video.mp4'),
            autoPlay = False,
            src = 'assets/gaze.mp4',
            ),
        html.Hr(),
        html.P("this is outputted")
        ])

# creates the area of interest visualization
def create_aoi(start, duration, files):
    number_of_files = 0
    dataf = []
    videof = []
    final_array = [] # store the frame arrays for each file
    for file in files:
        dataf.append(pd.read_csv('data/' + file + '/eye_track_2D_hit.tsv', delimiter= '\t'))
        videof.append(pd.read_csv('data/' + file + '/video_events.tsv', delimiter= '\t'))
        number_of_files = number_of_files + 1
    print("create aoi triggered")
    print(start, " start")
    print(duration, " duration")
    print(number_of_files, " this is number of files")
    #print(board, " board")
    video = cv2.VideoCapture('https://vel.engr.uga.edu/apps/VRDEO/raydiagrams.mp4')
    f_width = int(video.get(3))
    f_height = int(video.get(4))
    fps = int(video.get((cv2.CAP_PROP_FPS)))
    output = cv2.VideoWriter('assets/aoi.mp4', cv2.VideoWriter_fourcc(*'mpv4'),fps,(f_width,f_height))#*'mp4v'

    colon = start.find(':')
    minute = int(start[0:colon])
    seconds = int(start[colon + 1:len(start)])
    seconds = float(seconds + minute * 60)
    #second_dateTime = pd.to_datetime(seconds, format='%S')
    #duration_dateTime = pd.to_datetime(duration, format='%S')
    video.set(cv2.CAP_PROP_POS_MSEC,seconds * 1000)

    check, frame = video.read()
    rectangle = cv2.selectROI(frame, fromCenter=False)
    x0 = rectangle[0]
    y0 = rectangle[1]
    x1 = rectangle[2] + rectangle[0]
    y1 = rectangle[3] + rectangle[1]

    i = 0

    frame_array = []

    cv2.destroyAllWindows()

    while (i < number_of_files):
        video.set(cv2.CAP_PROP_POS_MSEC,seconds * 1000)
        frame_array = []
        print(i, " this is i iteration in number of files")
        #video.set(cv2.CAP_PROP_POS_MSEC,seconds * 1000)
        x = list(dataf[i]['x'])
        y = list(dataf[i]['y'])
        eye_board = list(dataf[i]['board'])
        dfglobal = list(dataf[i]['global time'])
        dfglobal = pd.to_datetime(dfglobal, format='%M:%S.%f')

        vidglobal = list(videof[i]['global'])
        vidglobal = pd.to_datetime(vidglobal, format='%M:%S.%f')
        vid_time = list(videof[i]['vid time (sec)'])
        video_action = list(videof[i]['action'])

        in_video = []
        increment = 0
        globalx = []
        globaly = []
        globalvid = []
        for j in vid_time:
            #print(j, " this is j")
            if not math.isnan(j):
                if ((seconds <= j) and (seconds + duration >= j)):
                    in_video.append(increment)
                    #print(j, " this is j")
                elif (seconds > j and video_action[increment] == 'play' and increment + 1 < len(vid_time) and vid_time[increment + 1] > seconds):
                    in_video.append(increment)
                    print(j, " this is j in second if")
            else:
                #print("else section")
                if (video_action[increment] == 'play' and increment + 1 < len(vid_time) and vid_time[increment + 1] > seconds):
                    in_video.append(increment)
                    #print("special nan case")
            increment = increment + 1 

        print(len(in_video), " this is the number in video")
        start_time = []
        end_time = []
        current = []
        current_end = []
        start_vid_index = []
       
        for index in in_video:
            #print(index, " this is index")
            if math.isnan(vid_time[index]):
                vid_time[index] = 0
                print(vid_time[index], " vid index after added if was hit")
            if index - 1 not in in_video:
                if video_action[index] == 'set_time': #handle video actions
                    if (video_action[index + 1] == 'pause' and vid_time[index] == vid_time[index + 1]):
                        temp_vid = vid_time[index] - seconds
                        temp_start = vidglobal[index] - timedelta(seconds=temp_vid)
                        start_time.append(temp_start)
                        current.append(seconds)
                        start_vid_index.append(index)
                    #paused
                elif index + 1 not in in_video and video_action[index] == 'play':
                    print("the special play triggered")
                    temp_vid = seconds - vid_time[index]
                    temp_start = vidglobal[index] + timedelta(seconds=temp_vid)
                    start_time.append(temp_start)
                    current.append(seconds)
                    start_vid_index.append(index)

                    temp_vid = seconds + duration - vid_time[index]
                    temp_end = vidglobal[index] + timedelta(seconds=temp_vid)
                    end_time.append(temp_end)
                    current_end.append(seconds + duration)
                    print(temp_end, " this is temp end in play thing")
                elif video_action[index] == 'play':
                    temp_vid = seconds - vid_time[index]
                    temp_start = vidglobal[index] + timedelta(seconds=temp_vid)
                    start_time.append(temp_start)
                    current.append(seconds)
                    start_vid_index.append(index)
                elif video_action[index] == 'set_url':
                    start_time.append(vidglobal[index])
                    current.append(vid_time[index])
                    start_vid_index.append(index)
            elif (video_action[index] == 'set_time' and vid_time[index] != 'nan'):
                if (index + 1 in in_video and video_action[index + 1] == 'pause' and vid_time[index] != vid_time[index + 1]):
                    end_time.append(vidglobal[index + 1])
                    current_end.append(vid_time[index + 1])

                    start_time.append(vidglobal[index])
                    current.append(vid_time[index])
                    start_vid_index.append(index)
                    #print("end and start time append in change time ----------------------------------")
                elif (video_action[index + 1] == 'pause' and vid_time[index] != vid_time[index + 1]):
                    start_time.append(vidglobal[index])
                    current.append(vid_time[index])
                    start_vid_index.append(index)
            elif video_action[index] == 'set_url':
                start_time.append(vidglobal[index])
                current.append(vid_time[index])
                start_vid_index.append(index)
            elif (video_action[index] == 'play' and index + 1 not in in_video):
                temp_vid = seconds + duration - vid_time[index]
                temp_end = vidglobal[index] + timedelta(seconds=temp_vid)
                end_time.append(temp_end)
                current_end.append(seconds + duration)
            elif (video_action[index] == 'pause' and index + 1 >= len(vidglobal)):
                end_time.append(dfglobal[len(dfglobal) - 1])
                current_end.append(vid_time[index])
    
        print(len(start_time), " start time length")
        print(len(end_time), " end time length")
        cycles = []
        timer_array = []

        count = 0
        time_repeat = 0
        onVideo = False
        readFrame = False
        loop = 0
        count = 0
        nomatch = False
        alpha = 0.5
        time_repeat = []
        loop_array = []
        nomatch_array = []
        onVideo_array = []
        isPaused = []
        inROI = []

        # get the cycle for each time range
        for l in start_time:
            timer = l
            time_parse = timer.strftime('%M:%S.%f')
            time_parse = time_parse[:-5]
            timer = pd.to_datetime(time_parse, format='%M:%S.%f')
            timer_array.append(timer)
            cycle = find_time(timer, dfglobal)
            cycles.append(cycle)
            time_repeat.append(0)
            loop_array.append(0)
            nomatch_array.append(False)
            onVideo_array.append(False)
            isPaused.append(False)
            inROI.append(False)

        iteration = 0 #checks what iteration of start and end times we are on
        current_indices = []

        current_vid_time = seconds
        final = seconds + duration
        print(current_vid_time, " this is the start time")
        print(final, " this is the final time")

        #check, frame = video.read()
        #rectangle = cv2.selectROI(frame, fromCenter=False)
        readFrame = False
        begin = True
        #inc = 1
        xArray = []
        yArray = []
        while current_vid_time <= final:
            #print(current_vid_time, " this is current vid time")
            current_indices.clear()
            for h in range(len(current)):
                if (current[h] <= current_vid_time and current_end[h] >= current_vid_time):
                    current_indices.append(h)

            #print(cycle, " this is cycle")
            #print(timer, " timer")
            for h in current_indices:
                if timer_array[h] == dfglobal[cycles[h]]:
                    for k in range(3): # determine how many times the global time in eye_track_2D_hit is repeated in the 100 ms value
                        if cycles[h] + k >= len(dfglobal):
                            break
                        if (dfglobal[cycles[h]] == dfglobal[cycles[h] + k]):
                            time_repeat[h] += 1
                        else:
                            break
            #print(time_repeat, " this is time repeat!!!!!!!!!!!!!!!!")
                if time_repeat[h] == 1:
                    loop_array[h] = -1
                    nomatch_array[h] = False
                elif timer_array[h] != dfglobal[cycles[h]]:
                    loop_array[h] = -1
                    nomatch_array[h] = True
                elif time_repeat[h] == 2:
                    loop_array[h] = 1
                    nomatch_array[h] = False
                elif time_repeat[h] == 3:
                    loop_array[h] = 2
                    nomatch_array[h] = False

                if (vidglobal[start_vid_index[h]] <= dfglobal[cycles[h]]):
                    if video_action[start_vid_index[h]] == 'set_time': #handle video actions
                        if (video_action[start_vid_index[h] + 1] == 'pause'):
                            isPaused[h] = True
                        start_vid_index[h] = start_vid_index[h] + 1 # set time and pause always at same time so always increment after set time
                    elif video_action[start_vid_index[h]] == 'play':
                        isPaused[h] = False
                    start_vid_index[h] = start_vid_index[h] + 1
            while (count < 3):
                #print("count trigger ", count)
                if (begin):
                    readFrame = False
                    begin = False
                elif (True in isPaused):
                    readFrame = False
                    #print(readFrame, " this is readframe")
                    #print(readFrame, " this is readframe")
                else:
                    readFrame = True
                    #print(readFrame, " this is readframe")

                #print(readFrame, " this is readFrame")
                for h in current_indices:
                    if (eye_board[cycles[h]] == "whiteboard (video)"):
                        if (y[cycles[h]] >= .616 and y[cycles[h]] <= .95):
                            onVideo_array[h] = True
                        else:
                           onVideo_array[h] = False
                    #print(onVideo, " this is onvideo")
                        
                    if readFrame:
                        #print (count, " this is count in readFrame")
                        #print("readFrame is true")
                        if (onVideo_array[h]):
                            xval = (((x[cycles[h]] - 0) * f_width) / 1) + 0
                            yval = (((y[cycles[h]] - .616) * f_height) / .334) + 0
                            yval = f_height - yval # to account for different coordinate grid
                            if (xval >= x0 and xval <= x1 and yval <= y1 and yval >= y0):
                                inROI[h] = True
                                print(xval, " xval")
                                print(yval, " yval")
                                print(x0, " x0")
                                print(y0, " y0")
                                print(x1, " x1")
                                print(y1, " y1")
                            else:
                                inROI[h] = False
                            #print("check before add to arrays")
                            if (inROI[h]):
                                xArray.append(xval)
                                yArray.append(yval)
                                globalx.append(xval)
                                globaly.append(yval)
                                globalvid.append(current_vid_time)
                    else:
                        if isPaused[h]:
                           if (onVideo_array[h]):
                                xval = (((x[cycles[h]] - 0) * f_width) / 1) + 0
                                yval = (((y[cycles[h]] - .616) * f_height) / .334) + 0
                                yval = f_height - yval # to account for different coordinate grid
                                if (xval >= x0 and xval <= x1 and yval <= y1 and yval >= y0):
                                    inROI[h] = True
                                    print(xval, " xval")
                                    print(yval, " yval")
                                    print(x0, " x0")
                                    print(y0, " y0")
                                    print(x1, " x1")
                                    print(y1, " y1")
                                else:
                                    inROI[h] = False
                                #print("check before add to arrays")
                                if (inROI[h]):
                                    xArray.append(xval)
                                    yArray.append(yval)
                                    globalx.append(xval)
                                    globaly.append(yval)
                                    globalvid.append(current_vid_time)
                        elif True not in isPaused:
                            if (onVideo_array[h]):
                                xval = (((x[cycles[h]] - 0) * f_width) / 1) + 0
                                yval = (((y[cycles[h]] - .616) * f_height) / .334) + 0
                                yval = f_height - yval # to account for different coordinate grid
                                if (xval >= x0 and xval <= x1 and yval <= y1 and yval >= y0):
                                    inROI[h] = True
                                    print(xval, " xval")
                                    print(yval, " yval")
                                    print(x0, " x0")
                                    print(y0, " y0")
                                    print(x1, " x1")
                                    print(y1, " y1")
                                else:
                                    inROI[h] = False
                                #print("check before add to arrays")
                                if (inROI[h]):
                                    xArray.append(xval)
                                    yArray.append(yval)
                                    globalx.append(xval)
                                    globaly.append(yval)
                                    globalvid.append(current_vid_time)

                    loop_array[h] = loop_array[h] - 1
                    if (loop_array[h] >= 0): # if repeated eye track hit for global time
                        cycles[h] = cycles[h] + 1
                if readFrame:
                    #print("readFrame if before heatmap stuff")
                    display = cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2)
                    for n in range(len(xArray)):
                        display = cv2.circle(display, (int(xArray[n]),int(yArray[n])), 20, (0,0,255), -1)
                    frame_array.append(display)
                        #output.write(frame)
                        #cv2.imshow('Heatmap', frame)
                    check, frame = video.read()
                count = count + 1
            count = 0
                    #print(timer, " this is timer")
                    #print(dfglobal[cycle], " this is dfglobal")
            for h in current_indices:
                if (not nomatch_array[h]):
                    cycles[h] = cycles[h] + 1
                timer_array[h] = timer_array[h] + timedelta(milliseconds=100)
                time_repeat[h] = 0
            if (True not in isPaused):
                current_vid_time = current_vid_time + .100
            if current_vid_time > final:
                display = cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2)
                for n in range(len(xArray)):
                    display = cv2.circle(display, (int(xArray[n]),int(yArray[n])), 20, (0,0,255), -1)
                frame_array.append(display)
        final_array.append(frame_array)
        i = i + 1 # cycle through selected experiments
    print("before write to video")
    #combine_array = []
    for j in range(len(final_array[0])):
        #print(i, " this is i for range len final_array 0")
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        #print(j, " this is j")
        for i in range(len(final_array)):
            #print(i, " this is i")
            if i + 1 < len(final_array):
                frame = final_array[i][j]
                next = final_array[i + 1][j]
                fin = cv2.addWeighted(next, 0.5, frame, 0.5, 0)
                final_array[i + 1][j] = fin
                #output.write(frame)
                #cv2.imshow('Heatmap', frame)
            else:
                if (j + 1 < len(final_array[0])):
                    frame = final_array[i][j]
                    next = final_array[i][j + 1]
                    fin = cv2.addWeighted(next, 0.5, frame, 0.5, 0)
                    final_array[i][j + 1] = fin
                #output.write(final_array[i][j])
                #cv2.imshow('Heatmap', final_array[i][j])
        output.write(final_array[i][j])
        cv2.imshow('AOI', final_array[i][j])

    video.release()
    # closes video frames
    cv2.destroyAllWindows()
    dict = {'x': globalx, 'y': globaly, 'video time' : globalvid}
    aoi_df = pd.DataFrame(dict)
    aoi_df.to_csv('data/aoiData.csv', index = False)
    print("Create AOI End")
        #inROI = False
    return html.Div([
        html.Video(
            controls = True,
            id = 'aoi_player',
            #src = app.get_asset_url(filename + '_video.mp4'),
            autoPlay = False,
            src = 'assets/aoi.mp4',
            ),
        html.Hr(),
        dash_table.DataTable(
        id='aoi_data_table',
        columns=[{"name": i, "id": i} for i in aoi_df.columns],
        data=df.to_dict('records'),
        )
        
        ])

# generate a heatmap for the specified video time
def create_heatmap(start, duration, files):
    number_of_files = 0
    dataf = []
    videof = []
    final_array = [] # store the frame arrays for each file
    for file in files:
        dataf.append(pd.read_csv('data/' + file + '/eye_track_2D_hit.tsv', delimiter= '\t'))
        videof.append(pd.read_csv('data/' + file + '/video_events.tsv', delimiter= '\t'))
        number_of_files = number_of_files + 1
    print("create heatmap triggered")
    print(start, " start")
    print(duration, " duration")
    print(number_of_files, " this is number of files")
    #print(board, " board")
    video = cv2.VideoCapture('https://vel.engr.uga.edu/apps/VRDEO/raydiagrams.mp4')
    f_width = int(video.get(3))
    f_height = int(video.get(4))
    fps = int(video.get((cv2.CAP_PROP_FPS)))
    output = cv2.VideoWriter('assets/heatmap.mp4', cv2.VideoWriter_fourcc(*'mpv4'),fps,(f_width,f_height))#*'mp4v'

    colon = start.find(':')
    minute = int(start[0:colon])
    seconds = int(start[colon + 1:len(start)])
    seconds = float(seconds + minute * 60)
    #second_dateTime = pd.to_datetime(seconds, format='%S')
    #duration_dateTime = pd.to_datetime(duration, format='%S')

    i = 0

    frame_array = []
    while (i < number_of_files):
        heatmap = np.zeros((f_height,f_width,1), np.uint8)
        video.set(cv2.CAP_PROP_POS_MSEC,seconds * 1000)
        frame_array = []
        print(i, " this is i iteration in number of files")
        #video.set(cv2.CAP_PROP_POS_MSEC,seconds * 1000)
        x = list(dataf[i]['x'])
        y = list(dataf[i]['y'])
        eye_board = list(dataf[i]['board'])
        dfglobal = list(dataf[i]['global time'])
        dfglobal = pd.to_datetime(dfglobal, format='%M:%S.%f')

        vidglobal = list(videof[i]['global'])
        vidglobal = pd.to_datetime(vidglobal, format='%M:%S.%f')
        vid_time = list(videof[i]['vid time (sec)'])
        video_action = list(videof[i]['action'])

        in_video = []
        increment = 0
        for j in vid_time:
            #print(j, " this is j")
            if not math.isnan(j):
                if ((seconds <= j) and (seconds + duration >= j)):
                    in_video.append(increment)
                    #print(j, " this is j")
                elif (seconds > j and video_action[increment] == 'play' and increment + 1 < len(vid_time) and vid_time[increment + 1] > seconds):
                    in_video.append(increment)
                    print(j, " this is j in second if")
            else:
                #print("else section")
                if (video_action[increment] == 'play' and increment + 1 < len(vid_time) and vid_time[increment + 1] > seconds):
                    in_video.append(increment)
                    #print("special nan case")
            increment = increment + 1 

        print(len(in_video), " this is the number in video")
        start_time = []
        end_time = []
        current = []
        current_end = []
        start_vid_index = []
       
        for index in in_video:
            #print(index, " this is index")
            if math.isnan(vid_time[index]):
                vid_time[index] = 0
                print(vid_time[index], " vid index after added if was hit")
            if index - 1 not in in_video:
                if video_action[index] == 'set_time': #handle video actions
                    if (video_action[index + 1] == 'pause' and vid_time[index] == vid_time[index + 1]):
                        temp_vid = vid_time[index] - seconds
                        temp_start = vidglobal[index] - timedelta(seconds=temp_vid)
                        start_time.append(temp_start)
                        current.append(seconds)
                        start_vid_index.append(index)
                    #paused
                elif index + 1 not in in_video and video_action[index] == 'play':
                    print("the special play triggered")
                    temp_vid = seconds - vid_time[index]
                    temp_start = vidglobal[index] + timedelta(seconds=temp_vid)
                    start_time.append(temp_start)
                    current.append(seconds)
                    start_vid_index.append(index)

                    temp_vid = seconds + duration - vid_time[index]
                    temp_end = vidglobal[index] + timedelta(seconds=temp_vid)
                    end_time.append(temp_end)
                    current_end.append(seconds + duration)
                    print(temp_end, " this is temp end in play thing")
                elif video_action[index] == 'play':
                    temp_vid = seconds - vid_time[index]
                    temp_start = vidglobal[index] + timedelta(seconds=temp_vid)
                    start_time.append(temp_start)
                    current.append(seconds)
                    start_vid_index.append(index)
                elif video_action[index] == 'set_url':
                    start_time.append(vidglobal[index])
                    current.append(vid_time[index])
                    start_vid_index.append(index)
            elif (video_action[index] == 'set_time' and vid_time[index] != 'nan'):
                if (index + 1 in in_video and video_action[index + 1] == 'pause' and vid_time[index] != vid_time[index + 1]):
                    end_time.append(vidglobal[index + 1])
                    current_end.append(vid_time[index + 1])

                    start_time.append(vidglobal[index])
                    current.append(vid_time[index])
                    start_vid_index.append(index)
                elif (video_action[index + 1] == 'pause' and vid_time[index] != vid_time[index + 1]):
                    start_time.append(vidglobal[index])
                    current.append(vid_time[index])
                    start_vid_index.append(index)
            elif video_action[index] == 'set_url':
                start_time.append(vidglobal[index])
                current.append(vid_time[index])
                start_vid_index.append(index)
            elif (video_action[index] == 'play' and index + 1 not in in_video):
                temp_vid = seconds + duration - vid_time[index]
                temp_end = vidglobal[index] + timedelta(seconds=temp_vid)
                end_time.append(temp_end)
                current_end.append(seconds + duration)
            elif (video_action[index] == 'pause' and index + 1 >= len(vidglobal)):
                end_time.append(dfglobal[len(dfglobal) - 1])
                current_end.append(vid_time[index])
    
        print(len(start_time), " start time length")
        print(len(end_time), " end time length")
        cycles = []
        timer_array = []

        count = 0
        time_repeat = 0
        onVideo = False
        readFrame = False
        loop = 0
        count = 0
        nomatch = False
        alpha = 0.5
        time_repeat = []
        loop_array = []
        nomatch_array = []
        onVideo_array = []
        isPaused = []

        # get the cycle for each time range
        for l in start_time:
            timer = l
            time_parse = timer.strftime('%M:%S.%f')
            time_parse = time_parse[:-5]
            timer = pd.to_datetime(time_parse, format='%M:%S.%f')
            timer_array.append(timer)
            cycle = find_time(timer, dfglobal)
            cycles.append(cycle)
            time_repeat.append(0)
            loop_array.append(0)
            nomatch_array.append(False)
            onVideo_array.append(False)
            isPaused.append(False)

        iteration = 0 #checks what iteration of start and end times we are on
        current_indices = []

        current_vid_time = seconds
        final = seconds + duration
        print(current_vid_time, " this is the start time")
        print(final, " this is the final time")

        check, frame = video.read()
        readFrame = False
        begin = True
        #inc = 1
        xArray = []
        yArray = []
        while current_vid_time <= final:
            #print(current_vid_time, " this is current vid time")
            current_indices.clear()
            for h in range(len(current)):
                if (current[h] <= current_vid_time and current_end[h] >= current_vid_time):
                    current_indices.append(h)

            #print(cycle, " this is cycle")
            #print(timer, " timer")
            for h in current_indices:
                if timer_array[h] == dfglobal[cycles[h]]:
                    for k in range(3): # determine how many times the global time in eye_track_2D_hit is repeated in the 100 ms value
                        if cycles[h] + k >= len(dfglobal):
                            break
                        if (dfglobal[cycles[h]] == dfglobal[cycles[h] + k]):
                            time_repeat[h] += 1
                        else:
                            break
                if time_repeat[h] == 1:
                    loop_array[h] = -1
                    nomatch_array[h] = False
                elif timer_array[h] != dfglobal[cycles[h]]:
                    loop_array[h] = -1
                    nomatch_array[h] = True
                elif time_repeat[h] == 2:
                    loop_array[h] = 1
                    nomatch_array[h] = False
                elif time_repeat[h] == 3:
                    loop_array[h] = 2
                    nomatch_array[h] = False

                if (vidglobal[start_vid_index[h]] <= dfglobal[cycles[h]]):
                    if video_action[start_vid_index[h]] == 'set_time': #handle video actions
                        if (video_action[start_vid_index[h] + 1] == 'pause'):
                            isPaused[h] = True
                        start_vid_index[h] = start_vid_index[h] + 1 # set time and pause always at same time so always increment after set time
                    elif video_action[start_vid_index[h]] == 'play':
                        isPaused[h] = False
                    start_vid_index[h] = start_vid_index[h] + 1
            while (count < 3):
                #print("count trigger ", count)
                if (begin):
                    readFrame = False
                    begin = False
                elif (True in isPaused):
                    readFrame = False
                    #print(readFrame, " this is readframe")
                    #print(readFrame, " this is readframe")
                else:
                    readFrame = True
                    #print(readFrame, " this is readframe")

                #print(readFrame, " this is readFrame")
                for h in current_indices:
                    if (eye_board[cycles[h]] == "whiteboard (video)"):
                        if (y[cycles[h]] >= .616 and y[cycles[h]] <= .95):
                            onVideo_array[h] = True
                        else:
                           onVideo_array[h] = False
                    #print(onVideo, " this is onvideo")

                    if readFrame:
                        #print (count, " this is count in readFrame")
                        #print("readFrame is true")
                        if (onVideo_array[h]):
                            #print("read frame true on video")
                            #print(onVideo, " add point to heatmap")
                            xval = (((x[cycles[h]] - 0) * f_width) / 1) + 0
                            yval = (((y[cycles[h]] - .616) * f_height) / .334) + 0
                            yval = f_height - yval # to account for different coordinate grid
                            #print("check before add to arrays")
                            xArray.append(xval)
                            yArray.append(yval)
                            #cv2.circle(heatmap, (int(xval), int(yval)), 20, (255,0,0), -1)
                    else:
                        if isPaused[h]:
                           if (onVideo_array[h]):
                            #print(onVideo, " add point to heatmap")
                                #print("read frame false is paused on video")
                                xval = (((x[cycles[h]] - 0) * f_width) / 1) + 0
                                yval = (((y[cycles[h]] - .616) * f_height) / .334) + 0
                                yval = f_height - yval # to account for different coordinate grid
                                xArray.append(xval)
                                yArray.append(yval)
                                #cv2.circle(heatmap, (int(xval), int(yval)), 20, (255,0,0), -1)
                        elif True not in isPaused:
                            if (onVideo_array[h]):
                                #print("read frame false not paused on video")
                                #print(onVideo, " add point to heatmap")
                                xval = (((x[cycles[h]] - 0) * f_width) / 1) + 0
                                yval = (((y[cycles[h]] - .616) * f_height) / .334) + 0
                                yval = f_height - yval # to account for different coordinate grid
                                xArray.append(xval)
                                yArray.append(yval)
                                #cv2.circle(heatmap, (int(xval), int(yval)), 20, (255,0,0), -1)

                    loop_array[h] = loop_array[h] - 1
                    if (loop_array[h] >= 0): # if repeated eye track hit for global time
                        cycles[h] = cycles[h] + 1
                if readFrame:
                    #print("readFrame if before heatmap stuff")
                    for n in range(len(xArray)):
                        cv2.circle(heatmap, (int(xArray[n]), int(yArray[n])), 20, (255,0,0), -1)
                    if (len(xArray) > 0): # may be able to exclude
                        heatmap = cv2.distanceTransform(heatmap, cv2.DIST_L2, 5)
                        #print("is this occuring")
                        heatmap = heatmap * 3.5
                        heatmap = np.uint8(heatmap)
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        frame = cv2.addWeighted(heatmap, 0.5, frame, 0.5, 0)
                        heatmap = np.zeros((f_height,f_width,1), np.uint8) # reset heatmap
                        #print("how about this")
                    frame_array.append(frame)
                        #output.write(frame)
                        #cv2.imshow('Heatmap', frame)
                    check, frame = video.read()
                count = count + 1
            count = 0
                    #print(timer, " this is timer")
                    #print(dfglobal[cycle], " this is dfglobal")
            for h in current_indices:
                if (not nomatch_array[h]):
                    cycles[h] = cycles[h] + 1
                timer_array[h] = timer_array[h] + timedelta(milliseconds=100)
                time_repeat[h] = 0
            if (True not in isPaused):
                current_vid_time = current_vid_time + .100
            if current_vid_time > final:
                for n in range(len(xArray)):
                    cv2.circle(heatmap, (int(xArray[n]), int(yArray[n])), 20, (255,0,0), -1)
                heatmap = cv2.distanceTransform(heatmap, cv2.DIST_L2, 5)
                heatmap = heatmap * 3.5
                heatmap = np.uint8(heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                frame = cv2.addWeighted(heatmap, 0.5, frame, 0.5, 0)
                frame_array.append(frame)
                #output.write(frame)
                #cv2.imshow('Heatmap', frame)
        final_array.append(frame_array)
        #frame_array.clear()
        #for g in range(len(final_array[0])):
        #     print(g, " this is g range")
        #print("just cleared frame array")
        i = i + 1 # cycle through selected experiments
    print("before write to video")
    #combine_array = []
    for j in range(len(final_array[0])):
        #print(i, " this is i for range len final_array 0")
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        #print(j, " this is j")
        for i in range(len(final_array)):
            #print(i, " this is i")
            if i + 1 < len(final_array):
                frame = final_array[i][j]
                next = final_array[i + 1][j]
                fin = cv2.addWeighted(next, 0.5, frame, 0.5, 0)
                final_array[i + 1][j] = fin
                #output.write(frame)
                #cv2.imshow('Heatmap', frame)
            else:
                if (j + 1 < len(final_array[0])):
                    frame = final_array[i][j]
                    next = final_array[i][j + 1]
                    fin = cv2.addWeighted(next, 0.5, frame, 0.5, 0)
                    final_array[i][j + 1] = fin
                #output.write(final_array[i][j])
                #cv2.imshow('Heatmap', final_array[i][j])
        output.write(final_array[i][j])
        cv2.imshow('Heatmap', final_array[i][j])

    video.release()
    # closes video frames
    cv2.destroyAllWindows()
    print("Create Heatmap End")
    return html.Div([
        html.Video(
            controls = True,
            id = 'heatmap_player',
            #src = app.get_asset_url(filename + '_video.mp4'),
            autoPlay = False,
            src = 'assets/heatmap.mp4',
            ),
        html.Hr(),
        html.P("this is outputted")
        ])

def find_time(time, listing):
    index = 0
    for i in listing:
        if time <= i:
            return index
        index = index + 1
    return 0

def find_min_index(time):
    min = time[0]
    for i in time:
        if i < min:
            min = i

def find_max_index(time):
    max = time[0]
    for i in time:
        if i > max:
            max = i

# calls processvideo to create a full experiment video
def create_experiment_video(filename):
    print("create_experiment_video")
    vidf = pd.read_csv('data/' + filename + '/video_events.tsv', delimiter= '\t')
    vidEntry = vidf['video'][0] # use the video from the video_events file
    video = cv2.VideoCapture(vidEntry)
    vidProc = ProcessVideo(video,filename)
    vidProc.get_video()
    del vidProc

def generate_datatable(value):
     try:
         df = pd.read_csv(value, delimiter= '\t')
     except Exception as e:
         print(e)
         return html.Div([
            'There was an error processing a required file.'
    ])

     return dash_table.DataTable(
         id='exp_data_table',
         columns=[{"name": i, "id": i} for i in df.columns],
         data=df.to_dict('records'),
         )

 # if only one file is selected, show file specific info
def generate_solo(value):
    value = value[0]
    file_exists = exists('assets/' + value + '_full_experiment.mp4')
    if file_exists:
        video = cv2.VideoCapture('assets/' + value + '_full_experiment.mp4')
        frame_num = video.get(cv2.CAP_PROP_FRAME_COUNT)
        print(frame_num, " this is frame num")
        fps = int(video.get(cv2.CAP_PROP_FPS))
        print(fps, " this is fps")
        vid_secs = int(frame_num/fps)
        return html.Div([
            html.H4('Generate Full Experiment Video'),
            dbc.Button('Generate', id='submit_experiment', n_clicks=0),
            html.Div(id='container-button-experiment',
                         children='Enter a value and press submit'),
            dash_player.DashPlayer(
                width="768px", height="432px",
                controls = True,
                id = 'vid_player',
                #src = app.get_asset_url(filename + '_video.mp4'),
                url = '/assets/' + value + '_full_experiment.mp4',
                ),

            dbc.Button('Download Experiment Video', id='experiment_btn', color='danger'),
            Download(id='download_experiment'),

            html.Hr(),

            html.Div([
                dcc.Slider(
                    id='vid_slider',
                    min=0,
                    max=vid_secs,
                    step=1,
                    marks={i: str(i) for i in [0, vid_secs]},
                    tooltip={"placement": "bottom", "always_visible": True},
                    #disabled = True,
                ),
                html.Div(id='slider_output_container')
            ]),

            html.Hr(),

            html.P("Update Interval for Current Video Time:", style={'margin-top': '30px'}),
            dcc.Slider(
                id='slider_interval',
                min=40,
                max=100,
                step=None,
                updatemode='drag',
                marks={i: str(i) for i in [40, 100]},
                value=100
            ),
            dcc.Dropdown(id='data_selector',
                options=[
                    {'label': 'canvas_switch', 'value' : 'data/' + value + '/canvas_switch.tsv'},
                    {'label': 'display_switch', 'value' : 'data/' + value + '/display_switch.tsv'},
                    {'label': 'eye_track_2D_hit', 'value' : 'data/' + value + '/eye_track_2D_hit.tsv'},
                    {'label': 'eye_tracking_objects', 'value' : 'data/' + value + '/eye_tracking_objects.tsv'},
                    {'label': 'video_events', 'value' : 'data/' + value + '/video_events.tsv'},
                    {'label': 'vrpen_packets', 'value' : 'data/' + value + '/vrpen_packets.tsv'}
                    ],
                multi=False,
                placeholder='Choose data to display',
                #style={'backgroundColor': '#1E1E1E'},
                className='data_selector')
            ])
    else:
        return  html.Div([
            html.H4('Generate Full Experiment Video'),
            dbc.Button('Generate', id='submit_experiment', n_clicks=0),
            html.Div(id='container-button-experiment',
                         children='Enter a value and press submit'),
            html.Hr(),
            #html.Div(id='full_experiment_output'),
            dcc.Dropdown(id='data_selector',
                options=[
                    {'label': 'canvas_switch', 'value' : 'data/' + value + '/canvas_switch.tsv'},
                    {'label': 'display_switch', 'value' : 'data/' + value + '/display_switch.tsv'},
                    {'label': 'eye_track_2D_hit', 'value' : 'data/' + value + '/eye_track_2D_hit.tsv'},
                    {'label': 'eye_tracking_objects', 'value' : 'data/' + value + '/eye_tracking_objects.tsv'},
                    {'label': 'video_events', 'value' : 'data/' + value + '/video_events.tsv'},
                    {'label': 'vrpen_packets', 'value' : 'data/' + value + '/vrpen_packets.tsv'}
                    ],
                multi=False,
                placeholder='Choose data to display',
                #style={'backgroundColor': '#1E1E1E'},
                className='data_selector')
            ])

# Serve videos in assets
@server.route('/assets/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(os.path.join(root_dir, 'assets'), path)


# app layout section

app.layout = html.Div([
    
    html.Div([
        html.H3('Video'),
        html.P('Enter video link for analysis'),
        html.Div(dbc.Input(id='main_vid_input', type='text', value='https://vel.engr.uga.edu/apps/VRDEO/raydiagrams.mp4', style={'width': '25%'})),
        dbc.Button('Submit URL', id='url_submit', n_clicks=0),
        dash_player.DashPlayer(
            width="768px", height="432px",
            controls = True,
            id = 'base_video',
            url = 'https://vel.engr.uga.edu/apps/VRDEO/raydiagrams.mp4',
        )]),

    html.Div(children=[
         html.P('Enter start time. Format: M:SS; Example: 1:45'),
         html.Div(dbc.Input(id='main_submit', type='text', style={'width': '25%'}))]),
    html.Div(children=[html.P('Enter duration for analysis. Example: 10'),
        html.Div(dbc.Input(id='video_duration', type='number', min='1', style={'width': '25%'}))]),
    dbc.Button('Start Analysis', id='submit_analysis', n_clicks=0),
    html.Div(id='container-for-main'),


    html.Div(id='output-placeholder'),
    html.Div(id='output-div'),
    html.Hr(),
    html.Div(id='output-datatable'),
])

# generate options for specific selected time-slice
def create_output_options(start, video_duration):
    file_exists = exists('data/aoiData.csv')
    checklist = []
    for filename in glob.glob('data/*.zip'):
        #print(filename, " print filename")
        temp = os.path.split(filename)
        #print(temp, "this is temp")
        checklist.append(temp[1])

    if file_exists:
        df = pd.read_csv('data/aoiData.csv')
        return html.Div(className = 'row', children = [
            html.P('Video Segment Selected: ' + start + ' plus ' + str(video_duration) + ' seconds.' ),
            html.Div(className='col', children = [
                html.Div(className = 'row', children = [
                    html.H4('Generate Gaze Points'),
                    dbc.Button('Submit', id='submit-gaze', n_clicks=0, color='success', style = {'width': '98%', 'margin':'10px'}),
                    html.Div(id='container-button-gaze',
                             children='Enter a value and press submit')
                ]),

                html.Div(className='row', children = [html.Video(
                        width="768", height="432",
                        controls = True,
                        id = 'gaze_player',
                        #src = app.get_asset_url(filename + '_video.mp4'),
                        autoPlay = False,
                        src = 'assets/gaze.mp4',
                )]),

                dbc.Button('Download Gaze Video', id='gaze_btn', color='danger'),
                Download(id='download_gaze'),

                html.Hr(),
        
                html.Div(className='row', children = [
                    html.H4('Generate AOI'),
                    dbc.Button('Submit', id='submit-val', n_clicks=0, color='success', style = {'width': '98%', 'margin':'10px'}),
                    html.Div(id='container-button-basic',
                             children='Enter a value and press submit')
                ]),

                html.Div(className='row', children = [html.Video(
                        width="768", height="432",
                        controls = True,
                        id = 'aoi_player',
                        #src = app.get_asset_url(filename + '_video.mp4'),
                        autoPlay = False,
                        src = 'assets/aoi.mp4',
                )]),

                dash_table.DataTable(
                    id='aoi_table',
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=df.to_dict('records'),
                    page_size = 10
                    ),

                dbc.Button('Download AOI Video', id='aoi_btn', color='danger'),
                Download(id='download_aoi'),

                dbc.Button('Download AOI Data', id='aoidata_btn', color='danger'),
                Download(id='download_aoidata'),
                
                html.Hr(),

                html.Div(className='row', children = [
                    html.H4('Generate Heatmap'),
                    dbc.Button('Submit', id='submit-heatmap', n_clicks=0, color='success', style = {'width': '98%', 'margin':'10px'}),
                    html.Div(id='container-button-heatmap',
                             children='Enter a value and press submit')
                ]),

                html.Div(className='row', children = [html.Video(
                        width="768", height="432",
                        controls = True,
                        id = 'heatmap_player',
                        #src = app.get_asset_url(filename + '_video.mp4'),
                        autoPlay = False,
                        src = 'assets/heatmap.mp4',
                )]),

                dbc.Button('Download Heatmap Video', id='heatmap_btn', color='danger'),
                Download(id='download_heatmap'),

            ], style = {'text-align' : 'center',
                        'margin': '10px'}),
            html.Div(className='col', children = [
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '98%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    # Allow multiple files to be uploaded
                    accept=".zip",
                    multiple=True
                ),
                html.Div(className='div-for-dropdown',
                    children=[
                        dbc.Checklist(
                            id="all-checklist",
                            options=[{"label": "All", "value": "All"}],
                            value=["All"],
                            #labelStyle={"display": "inline-block"},
                        ),
                        dbc.Checklist(id='fileselector',
                            options=checklist,#get_options(os.listdir(path)),
                            value=checklist,
                            #multi=True,
                            #style={'backgroundColor': '#1E1E1E'},
                            className='fileselector')
                    ],
                   style={'color': '#1E1E1E'}),
                ], style = {'text-align' : 'center',
                        'margin': '10px'})
         ])
    else:
        return html.Div(className = 'row', children = [
            html.P('Video Segment Selected: ' + start + ' plus ' + str(video_duration) + ' seconds.' ),
            html.Div(className='col', children = [
                html.Div(className = 'row', children = [
                    html.H4('Generate Gaze Points'),
                    dbc.Button('Submit', id='submit-gaze', n_clicks=0, color='success', style = {'width': '98%', 'margin':'10px'}),
                    html.Div(id='container-button-gaze',
                             children='Enter a value and press submit')
                ]),

                html.Div(className='row', children = [html.Video(
                        width="768", height="432",
                        controls = True,
                        id = 'gaze_player',
                        #src = app.get_asset_url(filename + '_video.mp4'),
                        autoPlay = False,
                        src = 'assets/gaze.mp4',
                )]),
        
                dbc.Button('Download Gaze Video', id='gaze_btn', color='danger'),
                Download(id='download_gaze'),

                html.Hr(),

                html.Div(className='row', children = [
                    html.H4('Generate AOI'),
                    dbc.Button('Submit', id='submit-val', n_clicks=0, color='success', style = {'width': '98%', 'margin':'10px'}),
                    html.Div(id='container-button-basic',
                             children='Enter a value and press submit')
                ]),

                html.Div(className='row', children = [html.Video(
                        width="768", height="432",
                        controls = True,
                        id = 'aoi_player',
                        #src = app.get_asset_url(filename + '_video.mp4'),
                        autoPlay = False,
                        src = 'assets/aoi.mp4',
                )]),
               
                dbc.Button('Download AOI Video', id='aoi_btn', color='danger'),
                Download(id='download_aoi'),

                html.Hr(),

                html.Div(className='row', children = [
                    html.H4('Generate Heatmap'),
                    dbc.Button('Submit', id='submit-heatmap', n_clicks=0, color='success', style = {'width': '98%', 'margin':'10px'}),
                    html.Div(id='container-button-heatmap',
                             children='Enter a value and press submit')
                ]),

                html.Div(className='row', children = [html.Video(
                        width="768", height="432",
                        controls = True,
                        id = 'heatmap_player',
                        #src = app.get_asset_url(filename + '_video.mp4'),
                        autoPlay = False,
                        src = 'assets/heatmap.mp4',
                )]),

                dbc.Button('Download Heatmap Video', id='heatmap_btn', color='danger'),
                Download(id='download_heatmap'),

            ], style = {'text-align' : 'center',
                        'margin': '10px'}),
            html.Div(className='col', children = [
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '98%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    # Allow multiple files to be uploaded
                    accept=".zip",
                    multiple=True
                ),
                html.Div(className='div-for-dropdown',
                    children=[
                        dbc.Checklist(
                            id="all-checklist",
                            options=[{"label": "All", "value": "All"}],
                            value=["All"],
                            #labelStyle={"display": "inline-block"},
                        ),
                        dbc.Checklist(id='fileselector',
                            options=checklist,#get_options(os.listdir(path)),
                            value=checklist,
                            #multi=True,
                            #style={'backgroundColor': '#1E1E1E'},
                            className='fileselector')
                    ],
                   style={'color': '#1E1E1E'}),
                ], style = {'text-align' : 'center',
                        'margin': '10px'})
         ])

@app.callback(
    Output('base_video', 'url'),
    Input('url_submit', 'n_clicks'),
    State('main_vid_input', 'value')    
)
def update_displayed_video(n_clicks, value):
    #print("triggered")
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if ('url_submit' in changed_id and value is not None):
        print(n_clicks, ' nclicks triggered')
        print(value, " value triggered")
        return value


@app.callback(
    Output("fileselector", "value"),
    Output("all-checklist", "value"),
    Input("fileselector", "value"),
    Input("all-checklist", "value"),
)
def sync_checklists(files_selected, all_selected): # code adapted from Dash docs advanced callbacks https://dash.plotly.com/advanced-callbacks
    checklist = []
    for h in glob.glob('data/*.zip'):
        temp = os.path.split(h)
        checklist.append(temp[1])
    ctx = dash.callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if input_id == "fileselector":
        all_selected = ["All"] if set(files_selected) == set(checklist) else []
    else:
        files_selected = checklist if all_selected else []
    return files_selected, all_selected


@app.callback(
    Output('container-button-experiment', 'children'),
    Input('submit_experiment', 'n_clicks'),
    State('fileselector', 'value')
)
def generate_experiment(n_clicks, file):
    print("generate full experiment trigger")
    print(n_clicks, " this is n clicks")
    print(file, " this is filename")
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if ('submit_experiment' in changed_id and file is not None):
        filename = file[0]
        children = [
            create_experiment_video(filename)
            ]
        return children


@app.callback(
    Output('container-for-main', 'children'),
    Input('submit_analysis', 'n_clicks'),
    State('main_submit', 'value'),
    State('video_duration', 'value')
)
def start_analysis(n_clicks, start, video_duration):
    print(start, " start")
    print(video_duration, " vid duration")
    if start is not None and video_duration is not None:
        children = [
            create_output_options(start, video_duration)
            ]
        return children

@app.callback(
    Output('container-button-gaze', 'children'),
    Input('submit-gaze', 'n_clicks'),
    State('fileselector', 'value'),
    State('main_submit', 'value'),
    State('video_duration', 'value')
)
def update_gaze(n_clicks, file, input_for_gaze, gaze_duration):
    print(input_for_gaze, " start value in update_gaze")
    print(file, " file selector file")
    print(n_clicks, " nclicks in update_gaze")
    print(gaze_duration, " this is duration in update_gaze")
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if ('submit-gaze' in changed_id and input_for_gaze is not None and gaze_duration is not None and file is not None):
        children = [
            create_gaze(input_for_gaze, gaze_duration, file)
            ]
        return children

@app.callback(
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    State('fileselector', 'value'),
    State('main_submit', 'value'),
    State('video_duration', 'value')
)
def update_aoi(n_clicks, file, input_on_submit, duration):
    print(input_on_submit, " start value in update_aoi")
    print(file, " file selector file")
    print(n_clicks, " nclicks in update_aoi")
    print(duration, " this is duration in update_aoi")
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if ('submit-val' in changed_id and input_on_submit is not None and duration is not None and file is not None):
        children = [
            create_aoi(input_on_submit, duration, file)
            ]
        return children

@app.callback(
    Output('container-button-heatmap', 'children'),
    Input('submit-heatmap', 'n_clicks'),
    State('fileselector', 'value'),
    State('main_submit', 'value'),
    State('video_duration', 'value')
)
def update_heatmap(n_clicks, file, input_for_heatmap, heatmap_duration):
    print(input_for_heatmap, " start value in update_heatmap")
    print(file, " file selector file")
    print(n_clicks, " nclicks in update_heatmap")
    print(heatmap_duration, " this is duration in update_heatmap")
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if ('submit-heatmap' in changed_id and input_for_heatmap is not None and heatmap_duration is not None and file is not None):
        children = [
            create_heatmap(input_for_heatmap, heatmap_duration, file)
            ]
        return children

@app.callback(Output('vid_player', 'intervalCurrentTime'),
              [Input('slider_interval', 'value')])
def update_slider_interval(value):
    return value


@app.callback(Output('vid_slider', 'value'),
              Input('vid_player', 'currentTime'),
              State('fileselector', 'value'))
def update_slider_time(currentTime, file):
    filename = file[0]
    result = update_slider(currentTime, filename)
    return result

#@app.callback(Output('slider_output_container', 'value'),
#              Input('vid_slider', 'value'))
#def update_below_slider(value):
#    return 'Video time is at "{}" seconds'.format(value)

#@app.callback(Output('base_video', 'playing'),
#              Input('vid_player', 'currentTime'),
#              State('vid_player', 'playing'),
#              State('fileselector', 'value'))
#def update_play_status(currentTime, status, file):
#    print(status, ' this is play status')
#    if status:
#        return check_play(currentTime, file)
#    else:
#        return False

@app.callback(Output('base_video', 'playing'),
              Input('vid_player', 'playing'))
              #State('vid_player', 'playing'),
              #State('fileselector', 'value'))
def update_play_status(status):
    print(status, ' this is play status')
    if status:
        return True
    else:
        return False


@app.callback(Output('base_video', 'seekTo'),
              Input('vid_player', 'currentTime'),
              State('fileselector', 'value'))
def update_seek_status(currentTime, file):
    filename = file[0]
    seek = check_set_time(currentTime, filename)
    #print(seek, " this is seek")
    if (seek != -1):
        return seek

@app.callback(Output('output-div', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def upload_files(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
     #children = [
      #      parse_contents(c, n, d) for c, n, d in
       #     zip(list_of_contents, list_of_names, list_of_dates)]
        #return children
        for content, name, date in zip(list_of_contents, list_of_names, list_of_dates):
            content_type, content_string = content.split(',')
            content_decoded = base64.b64decode(content_string)
            zip_str = io.BytesIO(content_decoded)
            zip_obj = ZipFile(zip_str, 'r')
            children = [zip_obj.extractall('data/' + name)]
            return children


# Update the checklist dynamically
@app.callback(Output('fileselector', 'options'),
              Input('upload-data', 'filename'))
def update_checklist(filename):
    checklist = []
    for h in glob.glob('data/*.zip'):
        #print(h, " print filename")
        temp = os.path.split(h)
        #print(temp, "this is temp")
        checklist.append(temp[1])
    return [{'label': i, 'value': i} for i in checklist]

@app.callback(Output('output-placeholder', 'children'),
              Input('fileselector', 'value'))
def update_options(value):
    if value is not None and len(value) == 1:
        children = [
            generate_solo(value)
            ]
        return children

@app.callback(Output('output-datatable', 'children'),
              Input('data_selector', 'value'))
def show_data(value):
    if value is not None:
        children = [
            generate_datatable(value)
            ]
        return children

# download callbacks

@app.callback(
    Output("download_aoi", "data"),
    Input("aoi_btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_aoi(n_clicks):
    file_exists = exists('assets/aoi.mp4')
    if file_exists:
        return send_file("./assets/aoi.mp4")

@app.callback(
    Output("download_gaze", "data"),
    Input("gaze_btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_gaze(n_clicks):
    file_exists = exists('assets/gaze.mp4')
    if file_exists:
        return send_file("./assets/gaze.mp4")

@app.callback(
    Output("download_heatmap", "data"),
    Input("heatmap_btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_heatmap(n_clicks):
    file_exists = exists('assets/heatmap.mp4')
    if file_exists:
        return send_file("./assets/heatmap.mp4")

@app.callback(
    Output("download_experiment", "data"),
    Input("experiment_btn", "n_clicks"),
    State('fileselector', 'value'),
    prevent_initial_call=True,
)
def download_experiment(n_clicks, file):
    filename = file[0]
    print(filename, " filename in download_experiment")
    file_exists = exists('assets/' + filename + '_full_experiment.mp4')
    path = 'assets/' + filename + '_full_experiment.mp4'
    with ZipFile('experiment.zip','w', compression=zipfile.ZIP_DEFLATED) as zip:
        zip.write(path)
    if file_exists:
        return send_file('experiment.zip')

@app.callback(
    Output("download_aoidata", "data"),
    Input("aoidata_btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_aoidata(n_clicks):
    file_exists = exists('data/aoiData.csv')
    if file_exists:
        return send_file("./data/aoiData.csv")

if __name__ == '__main__':
    app.run_server(debug=True)