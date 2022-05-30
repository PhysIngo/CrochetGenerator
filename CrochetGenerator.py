#pyinstaller CrochetGenerator.py --clean --onedir -F --icon Icon.ico
#pyinstaller CrochetGenerator.py --clean --onefile -F --icon Icon.ico

#python -m pip install git+https://github.com/julienr/meshcut.git

import sys
import os.path
#sys.path.insert(0,'C:/ProgramData/Anaconda3/pkgs/')
#cwd = os.getcwd()
#os.chdir('C:/ProgramData/Anaconda3/pkgs/')

import numpy as np
import cv2
#import pyvista as pv
import math
from scipy import interpolate
from scipy import ndimage

from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure


from skimage import measure

import tkinter as tk
from tkinter import ttk, font
from tkinter import filedialog as fd
from tkinter import *

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

#import mayavi.mlab as mlab
#import itertools
#import utils
#import ply
import meshcut
LARGE_FONT = ("Verdana 12 bold")#('Helvetica', 12)
NORMAL_FONT = ("Verdana", 10)
COLUMN_LEFT = 4
COLUMN_RIGHT = 3
FIG_COLOR = '#cccccc'
BACK_COLOR = '#eeeeee'
BUT_COLOR = '#aaaaaa'
# ------------------------------------------------- #
# --------------- Start functions ----------------- #
# ------------------------------------------------- #


class Application(tk.Frame):
    

    def fcn_loadSTL(self):

        
        self.filename = fd.askopenfilename()
        self.saveMesh = mesh.Mesh.from_file(self.filename)

        self.filepath["text"] = self.filename
        
        fcn_refMesh(self)
        fcn_plotMesh(self)
        
    def fcn_rotateMesh(self):
            
        x = self.x;
        y = self.y;
        z = self.z;
        xpos = (np.amax(x)+np.amin(x))/2
        ypos = (np.amax(y)+np.amin(y))/2
        zpos = (np.amax(z)+np.amin(z))/2

        if self.variable.get()=="X-Axis":
            self.your_mesh.rotate([1,0,0], math.radians(self.scaler.get() ))
          
        elif self.variable.get()=="Y-Axis":
            self.your_mesh.rotate([0,1,0], math.radians(self.scaler.get() ))

        elif self.variable.get()=="Z-Axis":
            self.your_mesh.rotate([0,0,1], math.radians(self.scaler.get() ))

        xnew = self.your_mesh.x.reshape(-1)
        ynew = self.your_mesh.y.reshape(-1)
        znew = self.your_mesh.z.reshape(-1)
        
        fcn_plotMesh(self)
        
    def fcn_plotMeshPoints(self):
            
        x = self.your_mesh.x.reshape(-1);
        y = self.your_mesh.y.reshape(-1);
        z = self.your_mesh.z.reshape(-1);
        
        self.x = x;
        self.y = y;
        self.z = z;
        
        
        self.MeshInterpolate = float(self.e_interpolate.get())
        if (self.MeshInterpolate>0):
            fcn_interpPoints(self)

        self.PlotFlag = 2
        fcn_showSlices(self)

    
        #points = np.array([x, y, z]).T
        # points is a 3D numpy array (n_points, 3) coordinates of a sphere
        #cloud = pv.PolyData(points)
        #print(cloud)
        #cloud.plot()
        
    
    def fcn_refreshMesh(self):
        fcn_refMesh(self)

    def fcn_generateCrochet(self):

        self = fcn_showCrochet(self)
        self.CrochetStart = int(self.e_crochetStart.get())
        self.CrochetOffset = float(self.e_CrochetOffset.get())
        self.Rounds = int(self.e_Rounds.get())
        self.COM = int(self.cv_var.get())
        self.Debug = bool(int(self.cv_debug.get()))
        self.MeshInterpolate = float(self.e_interpolate.get())
        self.CloseKernel = int(self.e_closingKernel.get())
    
        try:
            your_mesh = self.your_mesh
        except:
            self.T.delete("1.0","end")
            self.T.insert(END, "Error, no .stl was loaded or mesh is corrupted!")
            return
        
        
        self.T.delete("1.0","end")
        self.T.insert(END, "Generating Crochet")

        dim = int(self.e_PixelSize.get())
        v = fcn_Mesh2Matrix(your_mesh,dim,self)
        print("Filled 3D-Mask")    

        crochet_start = self.CrochetStart
        crochet_rounds = self.Rounds
        crochet_size = 1
        crochet_length = 0
        crochet_round = crochet_start
        crochet_offset_start = self.CrochetOffset

        isPlot = False
        is3D = True
        isEasy = bool(int(self.cv_easy.get()))
        tmp = self.filename.split("/")
        tmp = tmp[-1]
        file = tmp.rsplit('.', 1)[0]
        allText = 'Crochet Patter: ' + file + '\n\n'
        
        myStruct = [ structtype() for n in range(crochet_rounds) ]
            
        stepV = dim/crochet_rounds;
        startV = stepV * crochet_offset_start

        crochet_polys = []
        crochet_dir = []

        crochet_round = np.zeros((crochet_rounds))
        crochet_round[0] = crochet_start
        n = 0
        kernel = np.ones((self.CloseKernel,self.CloseKernel),np.uint8)
        for sc in range(crochet_rounds):

            zVal = round(startV+stepV*n);
            sliceV = v[:,:,zVal];
            sliceV = cv2.morphologyEx(sliceV, cv2.MORPH_CLOSE, kernel)
            
            myStruct[sc].zVal = zVal
            # calcualte mid of slice
            mid = np.where(sliceV == 1)
            myStruct[sc].mid = np.mean(mid, axis=1)-((sliceV[0].size+1)/2)

            # calc direction and if it is ellipse
            myStruct[sc].dir = -1;
            if n>0 and bow_length>0:
                # calculate direction via previews mid val
                if not( ((myStruct[sc].mid[0]-myStruct[sc-1].mid[0])==0) and ((myStruct[sc].mid[1]-myStruct[sc-1].mid[1])==0) ):
                    direc = np.array([myStruct[sc].mid[0]-myStruct[sc-1].mid[0],myStruct[sc].mid[1]-myStruct[sc-1].mid[1] ])
                    myStruct[sc].dir = (math.atan2(direc[0],direc[1]))*180/math.pi

            # if cb Center of Mass is checked, then rotate the matrix and get a new slice
            if (self.COM==1):
                tmid = np.mean(mid, axis=1)
                cent = np.array([round(tmid[0]),round(tmid[1]),zVal])
                if (myStruct[sc].dir > -1):
                    ang = -myStruct[sc].dir
                    dist = -direc[0]*math.sin(ang)+direc[1]*math.cos(ang)
                    zstep = myStruct[sc].zVal-myStruct[sc-1].zVal
                    ang2 = math.atan(dist/zstep)*180/math.pi
                else:
                    ang = 0
                    ang2 = 0
                        
                newslice = fcn_SliceObject(v, ang, ang2, cent, self.Debug)
                sliceV = newslice

            if (self.Debug):
                print(myStruct[sc].mid)
                print(myStruct[sc].dir)

            # find the contours of the slice
            myStruct[sc].sliceV = sliceV
            r = sliceV
            contours = measure.find_contours(r, 0.5)
            try:
                contour = contours[0]
            except:
                continue
            myStruct[sc].contour = contour
            
            # find length of contour
            bow_length = fcn_calcBow(contour)
            myStruct[sc].bow_length = bow_length
            
            # calculate crochet length
            if n==0:
                crochet_length = bow_length/(crochet_start*2);
                
                # check if is ellipse
                dist = np.zeros(len(contour))
                for i in range(len(contour)):
                    dist[i] = (contour[i,0]-myStruct[sc].mid[0])*(contour[i,0]-myStruct[sc].mid[0]) + (contour[i,1]-myStruct[sc].mid[1])*(contour[i,1]-myStruct[sc].mid[1])
                print("Min. distance: "+str(np.amin(dist))+", max. distance: "+str(np.amax(dist)))
                ellipse = (np.amax(dist))/(np.amin(dist))
                if ellipse>1.5:
                    print("This is a elipse")
                    
            elif bow_length>0:
                crochet_round[n] = round(bow_length/crochet_length);
            
            print('Z index: ' + str(zVal) + ', Bow Length: ' + str(bow_length))
            
            n = n + 1

        # close the end of the crochet...
        last_val = crochet_round[crochet_round.size-1]
        while (last_val>1.5*crochet_start):
            crochet_round = np.append(crochet_round,round(last_val/2))
            last_val = crochet_round[crochet_round.size-1]
            #myStruct[sc].dir = np.append(myStruct[sc].dir,[0,0]);
            
        saved_crochet_round = crochet_round
        crochet_round = np.around((crochet_round))

        for i in range(crochet_round.size):
        #    
        #    % make the crocket easier by rendering the stiches to half and full of
        #    % the start magic ring
            if i>0:
                add = crochet_round[i]-crochet_round[i-1]
            else:
                add = 0
        #    end
            if (isEasy and i>0):
                flag = False
                neigh = np.array([0,1,-1,2,-2])
                for j in neigh:
                    if ((crochet_round[i]+j)%crochet_start==0):
                        crochet_round[i] = crochet_round[i]+j
                        flag = True
                        break
                    elif (((crochet_round[i]+j)*2)%crochet_start==0):
                        crochet_round[i] = crochet_round[i]+j
                        flag = True
                        break
                    
                if (~flag and abs(add)==1):
                    crochet_round[i] = crochet_round[i-1]

                add = crochet_round[i]-crochet_round[i-1]

            dirtext = ''
            
            text = ''
            if i==0:
                text = 'mc '+str(crochet_start)+' ('+str(crochet_start)+')'
            elif add>0:
        #        % ATTENTION: Here I need to add a triple inc for faster increase
        #        % if inc is to strong, make the slope smaller
                sc = crochet_round[i]-2*add
                if sc<0:
                    crochet_round[i] = crochet_round[i-1]*2
                    add = crochet_round[i]-crochet_round[i-1]
                    add = abs(add)
                    sc = crochet_round[i]-2*add
                    
                if sc==0:
                    text = str(add)+' inc ('+str(crochet_round[i])+')'+dirtext
                elif (sc%add==0):
                    multipl = sc/add;
                    text = '['+str(multipl)+' sc, 1 inc]x'+str(add)+' ('+str(crochet_round[i])+')'+dirtext
                elif (add%sc==0):
                    multipl = add/sc;
                    text = '['+str(multipl)+' inc, 1 sc]x'+str(sc)+' ('+str(crochet_round[i])+')'+dirtext
                else:
                    text = 'sc '+str(sc)+'/inc '+str(add)+' ('+str(crochet_round[i])+')'+dirtext

             
                if ((crochet_round[i]-add)<0):
                    text = text+' SOME ERROR!'
                    
            elif add<0:
        #        % ATTENTION: Here I need to add a triple inc for faster increase
        #        % if inc is to strong, make the slope smaller
                add = abs(add);
                sc = crochet_round[i]-2*add
                if sc<0:
                    crochet_round[i] = math.ceil(crochet_round[i-1]/2)
                    add = crochet_round[i]-crochet_round[i-1]
                    add = abs(add)
                    sc = crochet_round[i-1]-2*add
                    
                if sc==0:
                    text = str(add)+' dec ('+str(crochet_round[i])+')'+dirtext
                elif (sc%add==0):
                    multipl = sc/add;
                    text = '['+str(multipl)+' sc, 1 dec]x'+str(add)+' ('+str(crochet_round[i])+')'+dirtext
                elif (add%sc==0):
                    multipl = add/sc;
                    text = '['+str(multipl)+' dec, 1 sc]x'+str(sc)+' ('+str(crochet_round[i])+')'+dirtext
                else:
                    text = 'sc '+str(sc)+'/dec '+str(add)+' ('+str(crochet_round[i])+')'+dirtext

             
                if ((crochet_round[i]-add)<0):
                    text = text+' SOME ERROR!'
                    
            else:
                text = 'sc '+str(crochet_round[i])+' ('+str(crochet_round[i])+')'
            

            if i==self.Rounds:
                print('--- Additional rounds to close the object ---')
                allText = allText + '--- Additional rounds to close the object --- \n'
                
            print('Round '+str(i+1)+': '+text+'')
            allText = allText + 'Round '+str(i)+': '+text+'' + '\n'
        print('');
        allText = allText + '\n'
        print('In total '+str(sum(crochet_round))+' stiches');    
        allText = allText + 'In total '+str(sum(crochet_round))+' stiches'
        self.allText = allText
        self.myStruct = myStruct
        self.T.delete("1.0","end")
        #T.pack(side=tk.RIGHT)
        self.T.insert(END, allText)

        self.StichesPerRound = crochet_round


        

    def showCrochet(self):
        fcn_showCrochet(self)
        
    def showSlices(self):
        self.PlotFlag = 1
        fcn_showSlices(self)

    def showStiches(self):
        self.PlotFlag = 0
        fcn_showSlices(self)

    # ------------------------------------------------- #
    # -------------------- Start GUI ------------------ #
    # ------------------------------------------------- #

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        self.path = os.getcwd()
        self.first = True

    def plotWindow(self):
        
        if isPlot:
            filterig, ax = pyplot.subplots()
            ax.imshow(r, cmap=pyplot.cm.gray)

            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            
            ax.axis('image')
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.show()
                

    def create_widgets(self):
        self.CrochetStart = 6
        self.CrochetOffset = 0.1
        self.MeshInterpolate = 0.2
        self.Rounds = 24
        self.PixelSize = 30
        self.PlotFlag = 1
        self.CloseKernel = 5
        self.Debug = False
        self.filename = ''
        self.allText = 'Crochet not generated!'
        
# LEFT TOP        
        self.label = tk.Label(self)
        self.label["text"] = "Crochet Generator"
        self.label.grid(row=0,column=0,columnspan=COLUMN_LEFT+COLUMN_RIGHT,sticky="NESW",pady=20)
        self.label.config(font=LARGE_FONT)
        
        self.label = tk.Label(self)
        self.label["text"] = "by Ingo Hermann"
        self.label.grid(row=0,column=COLUMN_LEFT+COLUMN_RIGHT-1,sticky="NESW",pady=20)
        self.label.config(font=NORMAL_FONT)
# LEFT SIDE
        # stl and plot
        self.button = tk.Button(self)
        self.button["text"] = "Load .STL"
        self.button["command"] = self.fcn_loadSTL
        #self.button["border"] = "1"
        self.button["bg"] = BUT_COLOR
        self.button.grid(row=1,column=0,columnspan=COLUMN_LEFT,sticky="NESW")
        
        self.filepath = tk.Label(self)
        self.filepath["text"] = "no file loaded"
        self.filepath.grid(row=2,column=0,columnspan=COLUMN_LEFT,sticky="NESW")

        OptionList = ["X-Axis","Y-Axis","Z-Axis"]
        self.variable = tk.StringVar(self)
        self.variable.set(OptionList[0])
        #self.variable.grid(row=3,column=0,sticky="NESW")
        
        self.opt = tk.OptionMenu(self, self.variable, *OptionList)
        self.opt.config(font=NORMAL_FONT)
        self.opt["bg"] = BUT_COLOR
        self.opt.grid(row=7,column=0,rowspan=2,sticky="NESW")
        
        self.scaler = tk.Scale(self, from_=-180, to=180, resolution=0.1, orient=tk.HORIZONTAL, length=360)
        #, command=self.fcn_rotateMesh
        self.scaler.grid(row=7, rowspan=2,column=1,columnspan=2, sticky="NESW")
        
        self.update = tk.Button(self)
        self.update["text"] = "Update Mesh"
        self.update["command"] = self.fcn_rotateMesh
        self.update["bg"] = BUT_COLOR
        self.update.grid(row=7,column=3,sticky="NESW")

        self.refresh = tk.Button(self)
        self.refresh["text"] = "Refresh Mesh"
        self.refresh["command"] = self.fcn_refreshMesh
        self.refresh["bg"] = BUT_COLOR
        self.refresh.grid(row=8,column=3,sticky="NESW")
        
        self.meshpoints = tk.Button(self)
        self.meshpoints["text"] = "Show Mesh Points"
        self.meshpoints["command"] = self.fcn_plotMeshPoints
        self.meshpoints["bg"] = BUT_COLOR
        self.meshpoints.grid(row=6,column=0,sticky="NESW")

        self.label_interpolate = tk.Label(self)
        self.label_interpolate["text"] = "Point distance:"
        self.label_interpolate.grid(row=4,column=0,sticky="e")
        self.e_interpolate = tk.Entry(self)
        self.e_interpolate.insert(0,self.MeshInterpolate)
        self.e_interpolate.config(font=NORMAL_FONT,background=BACK_COLOR)
        self.e_interpolate.grid(row=5,column=0,sticky="NESW")
        
        self.label_closingKernel = tk.Label(self)
        self.label_closingKernel["text"] = "Kernel size:"
        self.label_closingKernel.grid(row=4,column=1,sticky="e")
        self.e_closingKernel = tk.Entry(self)
        self.e_closingKernel.insert(0,self.CloseKernel)
        self.e_closingKernel.config(font=NORMAL_FONT,background=BACK_COLOR)
        self.e_closingKernel.grid(row=5,column=1,sticky="NESW")


        #self.label_newArg = tk.Label(self)
        #self.label_newArg["text"] = "Dummy:"
        #self.label_newArg.grid(row=4,column=2,sticky="e")
        self.e_newArg = tk.Entry(self)
        self.e_newArg.insert(0,"")
        self.e_newArg.config(font=NORMAL_FONT,background=BACK_COLOR)
        self.e_newArg.grid(row=5,column=2,sticky="NESW")
        self.e_newArg.configure(state="disabled")

        #self.label_newArg = tk.Label(self)
        #self.label_newArg["text"] = "Dummy:"
        #self.label_newArg.grid(row=4,column=3,sticky="e")
        self.e_newArg = tk.Entry(self)
        self.e_newArg.insert(0,"")
        self.e_newArg.config(font=NORMAL_FONT,background=BACK_COLOR)
        self.e_newArg.grid(row=5,column=3,sticky="NESW")
        self.e_newArg.configure(state="disabled")
        

        self.fig_3D = Figure(figsize=(5,5), dpi=100)
        self.fig_3D.patch.set_facecolor(FIG_COLOR)
        self.canvas = FigureCanvasTkAgg(self.fig_3D, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=9,column=0,columnspan=COLUMN_LEFT,sticky="NESW")
        

# RIGHT SIDE
        # crochet generation
        self.button2 = tk.Button(self)
        self.button2["text"] = "Generate Crochet"
        self.button2["command"] = self.fcn_generateCrochet
        self.button2["bg"] = BUT_COLOR
        self.button2.grid(row=1,column=COLUMN_LEFT,columnspan=COLUMN_RIGHT,sticky="NESW")
        
        self.label_CrochetStart = tk.Label(self)
        self.label_CrochetStart["text"] = "Crochet Start Ring"
        self.label_CrochetStart.grid(row=2,column=COLUMN_LEFT,sticky="e")
        
        self.e_crochetStart = tk.Entry(self)
        self.e_crochetStart.insert(0,self.CrochetStart)
        self.e_crochetStart.config(font=NORMAL_FONT,background=BACK_COLOR)
        self.e_crochetStart.grid(row=2,column=COLUMN_LEFT+1,sticky="w")
        
        self.label_Rounds = tk.Label(self)
        self.label_Rounds["text"] = "Rounds"
        self.label_Rounds.grid(row=3,column=COLUMN_LEFT,sticky="e")
        
        self.e_Rounds = tk.Entry(self)
        self.e_Rounds.insert(0,self.Rounds)
        self.e_Rounds.config(font=NORMAL_FONT,background=BACK_COLOR)
        self.e_Rounds.grid(row=3,column=COLUMN_LEFT+1,sticky="w")

        self.label_CrochetOffset = tk.Label(self)
        self.label_CrochetOffset["text"] = "Crochet Start Position"
        self.label_CrochetOffset.grid(row=4,column=COLUMN_LEFT,sticky="e")
        
        self.e_CrochetOffset = tk.Entry(self)
        self.e_CrochetOffset.insert(0,self.CrochetOffset)
        self.e_CrochetOffset.config(font=NORMAL_FONT,background=BACK_COLOR)
        self.e_CrochetOffset.grid(row=4,column=COLUMN_LEFT+1,sticky="w")

        self.label_PixelSize = tk.Label(self)
        self.label_PixelSize["text"] = "Pixel Render"
        self.label_PixelSize.grid(row=5,column=COLUMN_LEFT,sticky="e")
        
        self.e_PixelSize = tk.Entry(self)
        self.e_PixelSize.insert(0,self.PixelSize)
        self.e_PixelSize.config(font=NORMAL_FONT,background=BACK_COLOR)
        self.e_PixelSize.grid(row=5,column=COLUMN_LEFT+1,sticky="w")
		
        self.cv_easy = tk.IntVar()
        self.cb_easy = Checkbutton(self, variable=self.cv_easy)
        self.cb_easy["text"] = "Easy Pattern"
        self.cb_easy.grid(row=6,column=COLUMN_LEFT,sticky="w")
        
        self.cv_var = tk.IntVar()
        self.cb_COM = Checkbutton(self, variable=self.cv_var)
        self.cb_COM["text"] = "Along Center of Mass"
        self.cb_COM.grid(row=6,column=COLUMN_LEFT+1,sticky="w")
		
        self.cv_debug = tk.IntVar()
        self.cb_debug = Checkbutton(self, variable=self.cv_debug)
        self.cb_debug["text"] = "Debug"
        self.cb_debug.grid(row=6,column=COLUMN_LEFT+2,sticky="w")
        
        self.ShowCrochet = tk.Button(self)
        self.ShowCrochet["text"] = "Show Crochet"
        self.ShowCrochet["command"] = self.showCrochet
        self.ShowCrochet["bg"] = BUT_COLOR
        self.ShowCrochet.grid(row=7,column=COLUMN_LEFT,sticky="NESW")

        self.ShowSlices = tk.Button(self)
        self.ShowSlices["text"] = "Show Slices"
        self.ShowSlices["command"] = self.showSlices
        self.ShowSlices["bg"] = BUT_COLOR
        self.ShowSlices.grid(row=7,column=COLUMN_LEFT+1,sticky="NESW")
        
        self.ShowStiches = tk.Button(self)
        self.ShowStiches["text"] = "Show Stiches"
        self.ShowStiches["command"] = self.showStiches
        self.ShowStiches["bg"] = BUT_COLOR
        self.ShowStiches.grid(row=7,column=COLUMN_LEFT+2,sticky="NESW")
        
        self.T = tk.Text(self,width=62)
        self.T.insert(END,self.allText)
        self.T.config(font=NORMAL_FONT)
        self.T.grid(row=9,column=COLUMN_LEFT,columnspan=COLUMN_RIGHT,sticky="NESW")
        

def fcn_calcBow(polyn):
    bow_length = 0
    for i in range(polyn[:,0].size-1):
        bow_length = bow_length + ( (polyn[1+i,0]-polyn[i,0])*(polyn[1+i,0]-polyn[i,0]) + (polyn[1+i,1]-polyn[i,1])*(polyn[1+i,1]-polyn[i,1]) )

    return bow_length


    
def fcn_interpolatePoints(self, x, y, z, amount):
    newx = []
    newy = []
    newz = []
    if amount<1:
        amount = 1
        
    for i in range(len(x)-1):
                
        squareV = math.sqrt((x[i+1]-x[i])*(x[i+1]-x[i]) + (y[i+1]-y[i])*(y[i+1]-y[i]) +(z[i+1]-z[i])*(z[i+1]-z[i]));
        if (squareV>amount):
            newx = np.append(newx,x[i])
            newy = np.append(newy,y[i])
            newz = np.append(newz,z[i])
            for j in range(round(squareV/amount)):                
                newx = np.append(newx,(x[i+1]-x[i])*j/round(squareV/amount)+x[i] )
                newy = np.append(newy,(y[i+1]-y[i])*j/round(squareV/amount)+y[i] )
                newz = np.append(newz,(z[i+1]-z[i])*j/round(squareV/amount)+z[i] )
                
    newx = np.append(newx,x[len(x)-1])
    newy = np.append(newy,y[len(y)-1])
    newz = np.append(newz,z[len(z)-1])

    return newx, newy, newz
    
def fcn_refMesh(self):
    
    self.your_mesh = self.saveMesh
    self.x = self.your_mesh.x.reshape(-1)
    self.y = self.your_mesh.y.reshape(-1)
    self.z = self.your_mesh.z.reshape(-1)

    
    # norm mesh
    self.your_mesh.x = (self.your_mesh.x-np.amin(self.x))/(np.amax(self.x)-np.amin(self.x))*2-1
    self.your_mesh.y = (self.your_mesh.y-np.amin(self.y))/(np.amax(self.y)-np.amin(self.y))*2-1
    self.your_mesh.z = (self.your_mesh.z-np.amin(self.z))/(np.amax(self.z)-np.amin(self.z))*2-1
    self.x = self.your_mesh.x.reshape(-1)
    self.y = self.your_mesh.y.reshape(-1)
    self.z = self.your_mesh.z.reshape(-1)
    
    fcn_plotMesh(self)
    
def fcn_plotMesh(self):  
        
    axes = self.fig_3D.add_subplot(111, projection='3d') 
    axes.set_facecolor(FIG_COLOR)
    
    self.fig_3D.patch.set_facecolor(FIG_COLOR)
    scale = self.your_mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    pyplot.show()
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.your_mesh.vectors))


    axes.plot([0,0], [0,0], [-1.15,1.15], color='red', alpha=0.8, lw=4)

    legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Object', markerfacecolor='b', markersize=15),
            Line2D([0], [0], color='r', lw=4, label='Crochet direction')]

    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_zticks([])
    axes.legend(handles=legend_elements)
    
    self.canvas.draw()

def fcn_SliceObject(v_normal, ang, ang2, cent, debug):
    #z axis rotation
    old_cent = v_normal[cent[0],cent[1],cent[2]]
    v_normal[cent[0],cent[1],cent[2]] = 100
    v_rot = ndimage.interpolation.rotate(v_normal, angle=ang, axes=(0,1))
    v_rot = ndimage.interpolation.rotate(v_rot, angle=ang2, axes=(1,2))
    
    dim = np.array([len(v_normal),len(v_normal[0]),len(v_normal[0][0])])
    centers = np.array([cent[0],cent[1],cent[2]])

    
    coord = np.where(v_rot == np.amax(v_rot.ravel()))
    
    dim2 = np.array([len(v_rot),len(v_rot[0]),len(v_rot[0][0])])
    centers2 = np.array([coord[0][0],coord[1][0],coord[2][0]])

    if (debug):
        print(ang)
        print(ang2)
        print(dim)
        print(centers)
        print(dim2)
        print(centers2)
        
    # binarize the matricies
    v_normal[cent[0],cent[1],cent[2]] = old_cent

    x = np.array([])   
    y = np.array([])   
    z = np.array([])      
    for i in range(dim2[0]):
        for j in range(dim2[1]):
            for k in range(dim2[2]):
                if (v_rot[i,j,k]>0.5):
                    v_rot[i,j,k] = 1;
                    x=np.append(x,i)
                    y=np.append(y,j)
                    z=np.append(z,k)
                else:
                    v_rot[i,j,k] = 0;
                    
    x0 = np.array([])   
    y0 = np.array([])   
    z0 = np.array([])      
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                if (v_normal[i,j,k]>0.5):
                    x0=np.append(x0,i)
                    y0=np.append(y0,j)
                    z0=np.append(z0,k)
    #
    sliceV = v_rot[:,:,round(coord[2][0])]

    # plot 3D matrix
    if (debug):
        fig = pyplot.figure(figsize=pyplot.figaspect(0.5))
        ax = fig.add_subplot(1,2,1, projection='3d')
        ax.scatter(x0, y0, z0);
        
        axn = fig.add_subplot(1,2,2, projection='3d')
        axn.scatter(x, y, z);
        
        pyplot.show()
    
    return(sliceV)
	
	

class structtype():
    def __init__(self,**kwargs):
        self.Set(**kwargs)
    def Set(self,**kwargs):
        self.__dict__.update(kwargs)
    def SetAttr(self,lab,val):
        self.__dict__[lab] = val

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
        
def fcn_showCrochet(self):
    
    if hasattr(self, 'scaleSlices'):
        if self.scaleSlices.winfo_exists():
            self.scaleSlices.destroy()
            
    try:#if self.fig_Slices.winfo_exists():
        self.fig_Slices.destroy()
    except:
        print("")
            
    try:# self.canvas_Slices.winfo_exists():
        self.canvas_Slices.destroy()
    except:
        print("")
         
    if hasattr(self, 'T'):
        if self.T.winfo_exists():
            self.T.destroy()
            
    self.T = tk.Text(self,width=62)
    self.T.insert(END,self.allText)
    self.T.config(font=NORMAL_FONT)
    self.T.grid(row=9,column=COLUMN_LEFT,columnspan=COLUMN_RIGHT,sticky="NESW")

    return(self)
def fcn_interpPoints(self):
    
    x,y,z = fcn_interpolatePoints(self,self.x,self.y,self.z,self.MeshInterpolate)
    self.x = x;
    self.y = y;
    self.z = z;
    
def fcn_Mesh2Matrix(your_mesh,dim, self):

    x = your_mesh.x.reshape(-1)
    y = your_mesh.y.reshape(-1)
    z = your_mesh.z.reshape(-1)
    
    self.x = x;
    self.y = y;
    self.z = z;
    
    fcn_interpPoints(self)
    x = self.x;
    y = self.y;
    z = self.z;
    
    xmax = np.amax(x);
    xmin = np.amin(x);
    ymax = np.amax(y);
    ymin = np.amin(y);
    zmax = np.amax(z);
    zmin = np.amin(z);
        
    x = (x-np.amin(x))/(np.amax(x)-np.amin(x))
    y = (y-np.amin(y))/(np.amax(y)-np.amin(y))
    z = (z-np.amin(z))/(np.amax(z)-np.amin(z))
    
        
    if (self.Debug):
        print(len(x))
        print(len(y))
        print(len(z))
        print(x)
        print(y)
        print(z)
    
    v = np.zeros((dim+1,dim+1,dim+1))
    inc = float((self.CloseKernel-1)/2)
    for i in range(x.size):
        v[int(round(inc+x[i]*(dim-2*inc)))][int(round(inc+y[i]*(dim-2*inc)))][int(round(z[i]*(dim)))] = 1

    return(v)
        
def fcn_showSlices(self):

    if hasattr(self, 'T'):
        if self.T.winfo_exists():
            self.T.destroy()

    flag = True
    if hasattr(self, 'scaleSlices'):
        if self.scaleSlices.winfo_exists():
            flag = False
            
    if flag:

        self.scaleSlices = tk.Scale(self, from_=0, to=self.Rounds, resolution=1, orient=tk.HORIZONTAL, length=self.Rounds)
        self.scaleSlices["command"] = fcn_plotSlices(self)
        self.scaleSlices.grid(row=8, column=COLUMN_LEFT, columnspan=COLUMN_RIGHT,sticky="NESW")
    
    
    #if hasattr(self, 'fig_Slices'):
    try:#if self.fig_Slices.winfo_exists():
        self.fig_Slices.destroy()
    except:
        pass
    #if hasattr(self, 'canvas_Slices'):
    try:# self.canvas_Slices.winfo_exists():
        self.canvas_Slices.destroy()
    except:
        pass

    self.fig_Slices = Figure(figsize=(5,5), dpi=100)
    self.fig_Slices.patch.set_facecolor(FIG_COLOR)
    self.canvas_Slices = FigureCanvasTkAgg(self.fig_Slices, self)
    self.canvas_Slices.draw()
    self.canvas_Slices.get_tk_widget().grid(row=9,column=COLUMN_LEFT,columnspan=COLUMN_RIGHT,sticky="NESW")

    fcn_plotSlices(self)
    return(self)


def fcn_plotSlices(self):
    
    try:
        if self.Debug:
            print("Plot Flag: ")
            print(self.PlotFlag)
        if (self.PlotFlag==1):
            actslice = int(self.scaleSlices.get())
            sliceV = self.myStruct[actslice].sliceV
            
            text = str(actslice) + '/' + str(self.Rounds) + ': z=' + str(self.myStruct[actslice].zVal) + ', bow-length:' + str(self.myStruct[actslice].bow_length)
            contour = self.myStruct[actslice].contour
            
            axes = self.fig_Slices.add_subplot(111)
            axes.set_facecolor(FIG_COLOR)
            axes.clear()  
            axes.imshow(sliceV, cmap=pyplot.cm.gray)
            try:
                axes.plot(contour[:, 1], contour[:, 0], linewidth=2)
            except:
                pass
                
            axes.axis('image')
            axes.set_xticks([])
            axes.set_yticks([])
            if (self.Debug):
                print(text)
            axes.set_title(text)
            pyplot.show()
        
            self.canvas_Slices.draw()
        elif (self.PlotFlag==0):
                               
                
            axes = self.fig_Slices.add_subplot(111, projection='3d') 
            axes.set_facecolor(FIG_COLOR)  
            try:
                StichesPerRound = self.StichesPerRound
                for i in range(self.Rounds):
                    tmp = self.myStruct[i].contour
                    x = np.arange(0, len(tmp))
                    #print(x)
                    x2 = np.linspace(0, len(tmp)-1, 50 )#((self.StichesPerRound[i]).astype(int)*5) 
                    #print(x2)
                    f = interpolate.interp1d(x, tmp[:,0])
                    #print(f(x2))
                    f2 = interpolate.interp1d(x, tmp[:,1])
                    #print(f2(x2))
                    tmp2 = np.concatenate( (f(x2), f2(x2)), axis=0)
                    #print(tmp2)
                    contour = np.transpose(np.reshape(tmp2, ( 2,len(f(x2)) ) ), (1,0) )
                    #print(contour)
                    try:
                        # and smooth interpolate the stiches
                        #contour[:,0] = smooth(contour[:,0],3)
                        #contour[:,1] = smooth(contour[:,1],3)
                        # 
                        idx = np.round(np.linspace(0, len(contour) - 1, num=(self.StichesPerRound[i]).astype(int), endpoint=False)).astype(int)
                        #print(idx)
                        axes.scatter(contour[idx, 1], contour[idx, 0], contour[idx, 0]*0+self.myStruct[i].zVal)
                    except:
                        a = 0
                        #axes.scatter(contour[idx, 1], contour[idx, 0], contour[idx, 0]*0+self.myStruct[i].zVal)
            except:
                pass
                
            pyplot.show()
        
            self.canvas_Slices.draw()
            

        elif (self.PlotFlag==2):
                               
                
            axes = self.fig_Slices.add_subplot(111, projection='3d') 
            axes.set_facecolor(FIG_COLOR)
            steps = round(self.x.size/1000)
            if steps<0:
                steps = 1
                
            if self.Debug:
                print("Steps/Size:")
                print(steps)
                print(self.x.size)
                
            for i in range(self.x.size):
                if (i%steps==0):    
                    axes.scatter(self.x[i], self.y[i], self.z[i])
                    
                
            pyplot.show()
        
            self.canvas_Slices.draw()
            

        #filterig, ax = pyplot.subplots()
        #ax.imshow(sliceV, cmap=pyplot.cm.gray)
        #ax.plot(contour[:, 1], contour[:, 0], linewidth=2)            
        #ax.axis('image')
        #ax.set_xticks([])
        #ax.set_yticks([])
        #pyplot.title(text)
        #pyplot.show()

                
    except:
        print("")

    # ------------------------------------------------- #
    # ---------------------- MAIN --------------------- #
    # ------------------------------------------------- #

class Window:
    def __init__(self, master):
        self.master = master
 
        style = ttk.Style()
        
        root.tk.call('source', 'azure dark/azure dark.tcl')
        style.theme_use('azure')
        style.configure("Accentbutton", foreground='white')
        style.configure("Togglebutton", foreground='white')
 
        style.theme_use("default")

 
 
root = tk.Tk()
window = Window(root)
#('clam', 'alt', 'default', 'classic')
#s=ttk.Style()
#s.theme_use('default')


root.title("Crochet Generator")
#try:
#    root.iconbitmap('Icon.ico')
#except:
#    print("no icon found!")
    
root.defaultFont = font.nametofont("TkDefaultFont")
root.defaultFont.configure(family="Verdana",size=10,weight=font.NORMAL)
app = Application(master=root)
app.mainloop()
