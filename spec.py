# import the opencv library
import cv2
from skimage import exposure
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt


# define a video capture object
#vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')




for i in range(1,11):
    frame = cv2.imread('yale/all/face'+str(i)+'.jpg') 
    blur = cv2.GaussianBlur(frame,(5,5),0)
    
    for j in range(1,20):
        test = cv2.imread('ressource/dataset/database15/face'+format(j, '02d')+'_01.jpg') 
        #blur2 = cv2.GaussianBlur(test,(5,5),0)
        #print(frame)
        matched = exposure.match_histograms(test, frame, channel_axis=-1)
        # src = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        # ref = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, 
        #                                 figsize=(8, 3),
        #                                 sharex=True, sharey=True)
        # ax1.imshow(src)
        # ax1.set_title('Source')
        # ax2.imshow(ref)
        # ax2.set_title('Reference')
        # ax3.imshow(matched)
        # ax3.set_title('Matched')

        # plt.tight_layout()
        # plt.show()
        #matched = cv2.cvtColor(matched, cv2.COLOR_BGR2GRAY)
       
        #cv2.imwrite("ressource/dataset/croppedfaces2/face"+format((i-1)//5 +1, '02d')+'_'+format((i-1)%5 +1, '02d')+".jpg", frame)
        #matched2 = cv2.equalizeHist(matched)
        #start = 15
       
        start = 5
        
        num = i +  start
        if j == 3 or j== 10 or j ==16:
            print("test "+str(j))
            cv2.imwrite('ressource/dataset/database15Enhanced/face'+format(j, '02d')+'_'+ format(num, '02d')+'.jpg',matched)

cv2.destroyAllWindows()
