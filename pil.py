from PIL import Image, ImageDraw,ImageFont
import cv2
import numpy as np
import math


def testPilDraw():

    img = cv2.imread(r"E:\code\pro_singleScanAuto\result\1130\1.jpg")
    Image1 = Image.fromarray(img)

    draw =ImageDraw.Draw(Image1)
    draw.line((20, 20, 150, 150), 'green')
    # draw.rectangle((100, 200, 300, 400), 'black', 'red')
    # draw.arc((100, 200, 300, 400), 0, 180, 'yellow')
    draw.ellipse((100, 100, 120, 120), 'blue', 'wheat')
    draw.point((100,100),'blue')
    filepath = 'simsun.ttc'
    font = ImageFont.truetype(filepath, 40)#设置字
    draw.text((100, 50), '对角线：', font = font,fill=(255,0,0,0))

    img = np.array(Image1)

    cv2.imwrite(r"E:\code\pro_singleScanAuto\result\1130\hels.jpg",img)


def testScalePil():
    print('hello')
    img = cv2.imread(r"E:\code\pro_singleScanAuto\result\1130\1.jpg")

    cv2.rectangle(img,(100,100),(300,300),(0,0,255),2)

    height_new = int(img.shape[0] / 2)

    width_new = int(img.shape[1] / 2)

    #img = cv2.resize(img,)

    Image1 = Image.fromarray(img)



    Image1 = Image1.resize((width_new,height_new), Image.ANTIALIAS)

    draw = ImageDraw.Draw(Image1)


    # Image1 = Image1.resize((300,300),Image.ANTIALIAS)

    # draw.line((20, 20, 150, 150), 'green')
    #
    # draw.ellipse((100, 100, 120, 120), 'blue', 'wheat')
    # draw.point((100, 100), 'blue')


    filepath = 'simsun.ttc'
    font = ImageFont.truetype(filepath, 20)  # 设置字

    scale_point1 = (100//2,100//2)
    scale_point2 = (300//2,300//2)
    mid_x = (scale_point2[0] - scale_point1[0])//2 + scale_point1[0]-50
    mid_y = scale_point1[1]-20

    scaled_x_coord = mid_x
    scaled_y_coord = mid_y

    draw.text((scaled_x_coord, scaled_y_coord), '水平距离：', font=font, fill=(0, 255, 0, 0))

    #Image1 = Image1.transpose(Image.ROTATE_270)  # 将图片旋转90度
    img = np.array(Image1)

    cv2.imwrite(r"E:\code\pro_singleScanAuto\result\1130\hels3.jpg", img)


def scaleRect():
    img = cv2.imread(r"E:\code\pro_singleScanAuto\result\1130\1.jpg")

    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 255), 2)
    cv2.imwrite(r"E:\code\pro_singleScanAuto\result\1130\hels3.jpg", img)

    Image1 = Image.fromarray(img)

    x_size = int(img.shape[0] / 2)
    y_size = int(img.shape[1] / 2)

    Image1 = Image1.resize((x_size, y_size), Image.ANTIALIAS)

    Image1.save(r"E:\code\pro_singleScanAuto\result\1130\hels4.jpg")

    Image1.show()


def scalePoints(points,scale):
    scalePts = []
    for pt in points:

        scale_x = int(pt[0]/scale)
        scale_y = int(pt[1]/scale)
        scalePts.append([scale_x,scale_y])
    return scalePts


def rotateText():
    canvas = Image.new('RGB',(500,500),(255,255,255))

    font = ImageFont.truetype('simsun.ttc',40)

    draw = ImageDraw.Draw(canvas)

    text = "hello world"

    position = (100,200)

    color = (0,0,0)

    draw.text(position,text,fill=color,font=font)

    angle = 40

    rotate_image = canvas.rotate(angle,expand=True)

    rotate_image.show()


def affineCoord(pt,matRotation):

    res = np.dot(matRotation, np.array([[pt[0]], [pt[1]], [1]]))

    res = res.reshape(1, 2)[0]

    res = np.int16(res)

    return res.tolist()


def affineImg(img,degree):

    height, width = img.shape[:2]

    heightNew = int(width * abs(math.sin(math.radians(degree))) + height * abs(math.cos(math.radians(degree))))

    widthNew = int(height * abs(math.sin(math.radians(degree))) + width * abs(math.cos(math.radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) // 2

    matRotation[1, 2] += (heightNew - height) // 2

    affine_img = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))

    return affine_img,matRotation


def generateImg():
    img = np.zeros(630000)
    img = img + 255
    img = img.reshape(300, 700,3)

    cv2.line(img,(100,100),(200,150),(255,0,255),2)

    return img


def imgshow(img):
    cv2.imshow('img', img)
    cv2.waitKey(6000)


def addAccelerate(index):
    sum = 0
    for i in range(index):
        sum +=index
    return sum

def feibonaqieshulie(index):
    if index == 0:
        return 0
    elif index ==1:
        return 1
    else:
        return feibonaqieshulie(index-2)+feibonaqieshulie(index-1)

def feiboLoop(index):
    res = 0
    if index ==1:
        return 1
    elif index == 2:
        return 2

    iter = 3
    first = 1
    second = 2
    while iter<=index:
        res = first + second
        first = second
        second = res
        iter += 1

    return res

def testRotate(img,angle):

    img = cv2.rotate(img,angle)

    for i in range(img):
        print(i)

    print("rotate over ")


def imgHandle():
    img = np.zeros(80000)
    # img = img.reshape((200,400,-1))
    img = np.reshape(img,(200,400))
    img = img+255
    print(img.shape)
    print(img)
    imgshow(img)

def testPerspective(img):
    img = img[0:24,40:-1]
    print(img)

def testLine():
    pt1 = [0, 0]
    pt2 = [0, 2]
    pts = np.array([pt1,pt2])
    output = cv2.fitLine(pts,cv2.DIST_L2,0,0.01,0.01)
    k = output[1]/output[0]
    b = output[3] -k*output[2]

    print(output)
    print(k)
    print(b)

def testNumpy():
    dis2 = np.array([255],np.uint8)
    dis1 = np.array([0],np.uint8)

    dis2 = dis2.tolist()
    dis1 = dis1.tolist()

    res = dis2[0] - dis1[0]
    print(dis2)
    print(res)

if __name__ == '__main__':

    # cnt = 190
    #
    # sum1 = feiboLoop(cnt)
    # print(sum1)

    testNumpy()

    # testPilDraw()

    # img = generateImg()
    #
    # img,matRotation = affineImg(img,120)
    #
    # start_pt = (100,100)
    #
    # end_pt = (200,150)
    #
    # pt1 = affineCoord(start_pt,matRotation)
    #
    # pt2 = affineCoord(end_pt, matRotation)
    #
    # cv2.line(img,tuple(pt1),tuple(pt2),(0,255,0),1)
    #
    # imgshow(img)

