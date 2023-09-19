import cv2
import face_recognition

img1 = face_recognition.load_image_file('Obama.jpg')
img1 = cv2.cvtColor(img1 , cv2.COLOR_BGR2RGB)

img1test= face_recognition.load_image_file('Obama2.jpg')
img1test = cv2.cvtColor(img1test, cv2.COLOR_BGR2RGB)

face  = face_recognition.face_locations(img1)[0]
encodeface = face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1,(face[3],face[0]),(face[1],face[2]),(0,255,0),3)

facetest = face_recognition.face_locations(img1test)[0]
encodefacetest = face_recognition.face_encodings(img1test)[0]
cv2.rectangle(img1test,(facetest[3],facetest[0]),(facetest[1],facetest[2]),(0,255,0),3)

res = face_recognition.compare_faces([encodeface],encodefacetest)
face_dis = face_recognition.face_distance([encodeface],encodefacetest)
print(res,face_dis)

cv2.putText(img1test,f"{res} {round(face_dis[0],2)}" , (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow("Obama",img1)
cv2.imshow("Obama Test",img1test)
cv2.waitKey()
cv2.destroyAllWindows()

