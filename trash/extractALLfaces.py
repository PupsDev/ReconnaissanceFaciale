# mat_contents = scipy.io.loadmat("allFaces.mat")
# faces = mat_contents['faces']
# m = int(mat_contents['m'])
# n = int(mat_contents['n'])
# nfaces = np.ndarray.flatten(mat_contents['nfaces'])

# print(len(faces[0]))
# print(nfaces)

# k = 0
# ind = 0
# for i in range(1,len(faces[0])+1):
#     img = faces[:,(i-1)]
#     # if i < k+nfaces[ind]:
#     #     img = img - faces[:,k]
#     # else:
#     #     k += nfaces[ind]
#     #     ind+=1
#     img = img.reshape(168,192).T
#     img =cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)
#     #print(img) 
#     cv2.imwrite("yale/all/face"+str(i)+".jpg",img)