import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import supportingFunctions as sf

cwd=os.getcwd()
tf.reset_default_graph()

subDirectory='04Jun_0356pm_5L_10K_50E_AG'
#%%Read the testing data from dataset.hdf5 file

noise_levels=[0.0,0.01,0.03,0.05]
results=[]

#%% Load existing model. Then do the reconstruction
modelDir = cwd + '/savedModels/' + subDirectory

loadChkPoint=tf.train.latest_checkpoint(modelDir)

config=tf.ConfigProto()
config.gpu_options.allow_growth=True

for sigma in noise_levels:

    print("\n==========================")
    print("Noise sigma =",sigma)
    print("==========================")

    tstOrg,tstAtb,tstCsm,tstMask=sf.getTestingData(sigma)

    tf.reset_default_graph()

    with tf.Session(config=config) as sess:

        new_saver=tf.train.import_meta_graph(modelDir+'/modelTst.meta')

        new_saver.restore(sess,loadChkPoint)

        graph=tf.get_default_graph()

        predT=graph.get_tensor_by_name('predTst:0')

        maskT=graph.get_tensor_by_name('mask:0')

        atbT=graph.get_tensor_by_name('atb:0')

        csmT=graph.get_tensor_by_name('csm:0')

        rec=sess.run(
            predT,
            feed_dict={
                atbT:tstAtb,
                maskT:tstMask,
                csmT:tstCsm
            }
        )

    rec=sf.r2c(rec.squeeze())

    normOrg=sf.normalize01(np.abs(tstOrg))

    normAtb=sf.normalize01(np.abs(sf.r2c(tstAtb)))

    normRec=sf.normalize01(np.abs(rec))

    psnrInput=sf.myPSNR(normOrg,normAtb)

    psnrRec=sf.myPSNR(normOrg,normRec)

    results.append([sigma,psnrInput,psnrRec])

    plt.figure(figsize=(12,4))

    plt.subplot(131)
    plt.imshow(normOrg,cmap='gray')
    plt.axis('off')
    plt.title("Original")

    plt.subplot(132)
    plt.imshow(normAtb,cmap='gray')
    plt.axis('off')
    plt.title("Input\nPSNR %.2f"%psnrInput)

    plt.subplot(133)
    plt.imshow(normRec,cmap='gray')
    plt.axis('off')
    plt.title("MoDL\nPSNR %.2f"%psnrRec)

    plt.savefig("Result_sigma_"+str(sigma)+".png",dpi=200)

    plt.show()

print("\nFinal Results\n")

print("Noise\tInput\tMoDL")

for r in results:
    print("%.2f\t%.2f\t%.2f"%(r[0],r[1],r[2]))
