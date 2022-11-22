import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from get_pre_P import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from CNN_DCNN import Model, get_data, get_US
from mpl_toolkits.axes_grid1 import make_axes_locatable

ckpt_dir="./model"
def evaluate(modelname):
    with tf.Graph().as_default():
        predata, label, mask = get_data(modelname)
        Mean, S = get_US(modelname)
        images = predata.reshape(-1,208,340,2)
        images = images.astype(np.float32)
        model = Model()
        y_pre = model.inference(images,keep_prob=1.0)
        accuracy = model.accuracy(y_pre,label,mask)
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess,ckpt_dir+"/model.ckpt")
            acc = sess.run(accuracy) 
            predict_result = sess.run(y_pre).reshape(-1,208,340,2)
            reshape_pre = (predict_result*mask)
            label = label*S+Mean
            pre = reshape_pre*S+Mean
            erro = np.abs(reshape_pre-label)/np.abs(label)
            return reshape_pre,label, erro

def main(argv=None):
    
    predict, true, error = evaluate(num)
    x1,y1,x2,y2,x3,y3 = get_xy(num)
    font1 = {'family' : 'Times New Roman',
                      'weight' : 'normal',
                      'size' : 15}
            
    cm = plt.cm.get_cmap('coolwarm')
    mylist1 = ['#0000FF','#436EEE','#7A67EE','#C1CDCD','#EEE0E5','#FFC1C1','#FF0000']
    cm1 = LinearSegmentedColormap.from_list('chaos',mylist1)
    _min1, _max1 = true.min(), true.max()

    plt.figure(1,figsize=(6,3))
    ax1 = plt.subplot(111)
    im1 = ax1.imshow(true[:,:,0],cmap =cm1,extent=(-0.2,1.5,-0.52,0.52),origin='lower')
    ax1.plot(x1,y1,c='w',linewidth=1)
    ax1.plot(x2,y2,c='w',linewidth=1)
    ax1.fill_between(x1,y1,y3,facecolor='w')
    ax1.fill_between(x3,y1,y3,facecolor='w')
    ax1.set_yticks([-0.4,-0.2,0.0,0.2,0.4])
    ax1.set_xticks([-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
    ax1.tick_params(labelsize=12)
    ax1.set_title("P_true")
    divider = make_axes_locatable(plt.gca())
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad="3%")
    cb1 = plt.colorbar(im1, cax=cax)
    plt.tight_layout()
    plt.savefig("p_true_%d.png"%num,dpi=800)
    plt.clf()
    
    plt.figure(2,figsize=(6,3))
    ax2 = plt.subplot(111)
    im2 = ax2.imshow(predict[:,:,0],cmap =cm1,extent=(-0.2,1.5,-0.52,0.52),origin='lower')
    ax2.plot(x1,y1,c='w',linewidth=1)
    ax2.plot(x2,y2,c='w',linewidth=1)
    ax2.fill_between(x1,y1,y3,facecolor='w')
    ax2.fill_between(x3,y1,y3,facecolor='w')
    ax2.set_yticks([-0.4,-0.2,0.0,0.2,0.4])
    ax2.set_xticks([-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
    ax2.tick_params(labelsize=12)
    ax2.set_title("P_predict")
    divider = make_axes_locatable(plt.gca())
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad="3%")
    cb2 = plt.colorbar(im2, cax=cax)
    plt.tight_layout()
    plt.savefig("p_pre_%d.png"%num,dpi=800)
    plt.clf()
    
    
    plt.figure(3,figsize=(6,3))
    ax3 = plt.subplot(111)
    im3 = ax3.imshow(error[:,:,0],cmap =cm1,extent=(-0.2,1.5,-0.52,0.52),origin='lower')
    im3.set_clim(-1,1)
    ax3.plot(x1,y1,c='w',linewidth=1)
    ax3.plot(x2,y2,c='w',linewidth=1)
    ax3.fill_between(x1,y1,y3,facecolor='w')
    ax3.fill_between(x3,y1,y3,facecolor='w')
    ax3.set_yticks([-0.4,-0.2,0.0,0.2,0.4])
    ax3.set_xticks([-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
    ax3.tick_params(labelsize=12)
    ax3.set_title("P_error")
    divider = make_axes_locatable(plt.gca())
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad="3%")
    cb3 = plt.colorbar(im3, cax=cax)
    plt.tight_layout()
    plt.savefig("p_error_%d.png"%num,dpi=800)
    plt.clf()
    
    plt.figure(4,figsize=(6,3))
    ax4 = plt.subplot(111)
    im4 = ax4.imshow(true[:,:,1],cmap =cm1,extent=(-0.2,1.5,-0.52,0.52),origin='lower')
    ax4.plot(x1,y1,c='w',linewidth=1)
    ax4.plot(x2,y2,c='w',linewidth=1)
    ax4.fill_between(x1,y1,y3,facecolor='w')
    ax4.fill_between(x3,y1,y3,facecolor='w')
    ax4.set_yticks([-0.4,-0.2,0.0,0.2,0.4])
    ax4.set_xticks([-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
    ax4.tick_params(labelsize=12)
    ax4.set_title("U_true")
    divider = make_axes_locatable(plt.gca())
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad="3%")
    cb4 = plt.colorbar(im4, cax=cax)
    plt.tight_layout()
    plt.savefig("u_true_%d.png"%num,dpi=800)
    plt.clf()
    
    plt.figure(5,figsize=(6,3))
    ax5 = plt.subplot(111)
    im5 = ax5.imshow(predict[:,:,1],cmap =cm1,extent=(-0.2,1.5,-0.52,0.52),origin='lower')
    ax5.plot(x1,y1,c='w',linewidth=1)
    ax5.plot(x2,y2,c='w',linewidth=1)
    ax5.fill_between(x1,y1,y3,facecolor='w')
    ax5.fill_between(x3,y1,y3,facecolor='w')
    ax5.set_yticks([-0.4,-0.2,0.0,0.2,0.4])
    ax5.set_xticks([-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
    ax5.tick_params(labelsize=12)
    ax5.set_title("U_predict")
    divider = make_axes_locatable(plt.gca())
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", size="5%", pad="3%")
    cb5 = plt.colorbar(im5, cax=cax)
    plt.tight_layout()
    plt.savefig("u_pre_%d.png"%num,dpi=800)
    plt.clf()
    
    
    plt.figure(6,figsize=(6,3))
    ax6 = plt.subplot(111)
    im6 = ax6.imshow(error[:,:,1],cmap =cm1,extent=(-0.2,1.5,-0.52,0.52),origin='lower')
    im6.set_clim(-1,1)
    ax6.plot(x1,y1,c='w',linewidth=1)
    ax6.plot(x2,y2,c='w',linewidth=1)
    ax6.fill_between(x1,y1,y3,facecolor='w')
    ax6.fill_between(x3,y1,y3,facecolor='w')
    ax6.set_yticks([-0.4,-0.2,0.0,0.2,0.4])
    ax6.set_xticks([-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
    ax6.tick_params(labelsize=12)
    ax6.set_title("U_error")
    divider = make_axes_locatable(plt.gca())
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes("right", size="5%", pad="3%")
    cb6 = plt.colorbar(im6, cax=cax)
    plt.tight_layout()
    plt.savefig("u_error_%d.png"%num,dpi=800)
    plt.clf()
    
if __name__ == '__main__':
    tf.app.run()