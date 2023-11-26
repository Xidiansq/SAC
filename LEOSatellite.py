import numpy as np
import math as m
from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs72
# // https://celestrak.com/NORAD

class satellite():
    def __init__(self, satnam):
        self.satnam = satnam

    def read_tle_file(self, TLE_PATH):
        satlist = []
        tle_str0 = []
        tle_str1 = []
        tle_str2 = []
        f = open(TLE_PATH)
        lines = f.readlines()
        y = [170, 176, 175, 169, 124, 105, 115, 162, 161]
        for i in range(len(lines) // 3):
            if lines[3 * i][0] == "I":
                # print(int(lines[i][8:11]))
                if int(lines[3 * i][8:11]) in y:
                    continue

                else:
                    satname = int(lines[3 * i][8:11])
                    satlist.append(satname)
                    tle_str0.append(lines[3 * i][8:11])
                    tle_str1.append(lines[3 * i + 1])
                    tle_str2.append(lines[3 * i + 2])

        return satlist, tle_str1, tle_str2


class beam():
    def __init__(self, SateBLH):
        self.SateBLH = SateBLH

    def createInitBeamCenterPos(self, satnum, numPointsof, SateBLH, numPoints, arcDist):
        beamCenterAlloc = []
        for i in range(numPoints):
            # 方位角设计
            cellid = satnum * 48 + numPointsof + i
            azimuth = (i + 1) * 360.0 / numPoints
            a = 6378137.0
            b = 6356752.3142
            f = 1.0 / 298.257223563
            alpha1 = azimuth * m.pi / 180.0
            sinAlpha1 = m.sin(alpha1)
            cosAlpha1 = m.cos(alpha1)
            tanU1 = (1 - f) * m.tan(SateBLH[0] * m.pi / 180.0)
            cosU1 = 1 / m.sqrt((1 + tanU1 * tanU1))
            sinU1 = tanU1 * cosU1
            sigma1 = m.atan2(tanU1, cosAlpha1)
            sinAlpha = cosU1 * sinAlpha1
            cosSqAlpha = 1 - sinAlpha * sinAlpha
            uSq = cosSqAlpha * (a * a - b * b) / (b * b)
            A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
            B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))

            sigma = arcDist / (b * A)
            sigmaP = 2 * m.pi
            sinSigma = m.sin(sigma)
            cosSigma = m.cos(sigma)
            cos2SigmaM = m.cos(2 * sigma1 + sigma)
            while (abs(sigma - sigmaP) > 1e-12):
                deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM) -
                                    B / 6 * cos2SigmaM * ( -3 + 4 * sinSigma * sinSigma) * (                                                         -3 + 4 * cos2SigmaM * cos2SigmaM)))
                sigmaP = sigma
                sigma = arcDist / (b * A) + deltaSigma
            tmp = sinU1 * sinSigma - cosU1 * cosSigma * cosAlpha1
            lat2 = m.atan(
                (sinU1 * cosSigma + cosU1 * sinSigma * cosAlpha1) / ((1 - f) * m.sqrt(sinAlpha * sinAlpha + tmp * tmp)))
            lambd = m.atan((sinSigma * sinAlpha1) / (cosU1 * cosSigma - sinU1 * sinSigma * cosAlpha1))
            C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
            L = lambd - (1 - C) * f * sinAlpha * (
                    sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)))
            pointPosition = GeographicToCartesianCoordinates(lat2 * 180 / m.pi, SateBLH[1] + L * 180 / m.pi, 0,"GRS80")
            beamCenterAlloc.append([self.SateBLH, cellid, pointPosition])
        return beamCenterAlloc

def sateposition(satlist, tle_str1, tle_str2, epoch, step):
    satnum = len(satlist)
    position = []
    velocity = []
    undersate = []
    SateBLH = []
    allocate = []
    SateId = []
    for i in range(satnum):
        time = (step + epoch * 100) * 0.001  # 假设每一个epoch有100个step,1个step为1ms(或者1m),可以修改
        hours = time // 3600
        minutes = (time % 3600) // 60
        seconds = (time - hours * 3600 - minutes * 60)
        satellite = twoline2rv(tle_str1[i], tle_str2[i], wgs72)
        position1, velocity1 = satellite.propagate(2020, 6, 29, hours, minutes, seconds)  # 时间可到秒,时间应该设置为可变x,y,z
        position1 = (position1[0] * 1000, position1[1] * 1000, position1[2] * 1000)  # km变成m
        position.append(position1)
        SateId.append([satlist[i], position1])
        velocity.append(velocity1)
        SateBLH1 = ConstructFromVector(position1[0], position1[1], position1[2],"GRS80")
        SateBLH.append(SateBLH1)
        underposition =GeographicToCartesianCoordinates(SateBLH1[0],SateBLH1[1],0,"GRS80")   # 星下点坐标
        undersate.append(underposition)
        allocate.extend(setInitBeamCenterPos(i, SateBLH1, "IRIDIUM"))
        # position为卫星位置，SateId为卫星编号+卫星位置，undersate为卫星星下点集合，allocate为卫星坐标+小区编号+波束中心
    return position, SateId, SateBLH, undersate, allocate


# 确定卫星下波束中心点(传入参数是卫星)
def setInitBeamCenterPos(i, SateBLH, type):
    if type == "IRIDIUM":
        # 波束的数量以及功率大小
        numPointsofC1 = 3
        numPointsofC2 = 9
        numPointsofC3 = 15
        numPointsofC4 = 21
        # 设定每一圈里中心距离orbit inclination : 86.402 degree, 1.508 rad
        maxArcDistFromSubPos = 40008080.0 / 20.0  # 3334km：单星覆盖arc长度
        arcDist1 = maxArcDistFromSubPos * 1 / 8
        arcDist2 = maxArcDistFromSubPos * 3 / 8
        arcDist3 = maxArcDistFromSubPos * 5 / 8
        arcDist4 = maxArcDistFromSubPos * 7 / 8
        sate = beam(SateBLH)
        # 设置每一层
        allocate1 = sate.createInitBeamCenterPos(i, 0, SateBLH, numPointsofC1, arcDist1)
        allocate2 = sate.createInitBeamCenterPos(i, numPointsofC1, SateBLH, numPointsofC2, arcDist2)
        allocate3 = sate.createInitBeamCenterPos(i, numPointsofC1 + numPointsofC2, SateBLH, numPointsofC3, arcDist3)
        allocate4 = sate.createInitBeamCenterPos(i, numPointsofC1 + numPointsofC2 + numPointsofC3, SateBLH,
                                                 numPointsofC4, arcDist4)
    return allocate1 + allocate2 + allocate3 + allocate4

def GeographicToCartesianCoordinates(latitude, longitude, altitude, sphType):
    latitudeRadians = 0.01745329 * latitude
    longitudeRadians = 0.01745329 * longitude
    # a: semi - major axisofearth
    # e: first eccentricity ofearth
    EARTH_RADIUS = 6371e3
    EARTH_SEMIMAJOR_AXIS = 6378137
    EARTH_GRS80_ECCENTRICITY = 0.0818191910428158
    EARTH_WGS84_ECCENTRICITY = 0.0818191908426215
    if sphType == "SPHERE":
        a = EARTH_RADIUS
        e = 0
    if sphType == "GRS80":
        a = EARTH_SEMIMAJOR_AXIS
        e = EARTH_GRS80_ECCENTRICITY
    else:  # if sphType == WGS84
        a = EARTH_SEMIMAJOR_AXIS
        e = EARTH_WGS84_ECCENTRICITY
    Rn = a / (m.sqrt(1 - pow(e, 2) * pow(m.sin(latitudeRadians), 2)))  # radius of  curvature
    x = (Rn + altitude) * m.cos(latitudeRadians) * m.cos(longitudeRadians)
    y = (Rn + altitude) * m.cos(latitudeRadians) * m.sin(longitudeRadians)
    z = ((1 - pow(e, 2)) * Rn + altitude) * m.sin(latitudeRadians)
    cartesianCoordinates = [x, y, z]
    return cartesianCoordinates


def ConstructFromVector (x,y,z,type):
    if type=="SPHERE":
        m_polarRadius = 6378137.0
    if type=="WGS84":
        m_polarRadius = 6356752.314245
    if type=="GRS80":
        m_polarRadius = 6356752.314103
    equatorRadius = 6378137.0
    m_equatorRadius = equatorRadius
    m_e2Param = ((m_equatorRadius * m_equatorRadius) - (m_polarRadius * m_polarRadius)) / (m_equatorRadius * m_equatorRadius )
    equatorRadius = 6378137.0
    # distance from the position point (P) to earth center point (origin O)
    op = m.sqrt ( x * x + y * y + z * z )

    if ( op > 0 ):
        # longitude calculation
        lon = m.atan (y / x)
        # scale longitude between - PI and PI (-180 and 180 in degrees)
        if ( x != 0) | (y != 0 ):
            m_longitude = m.atan (y / x ) * 180.0 / m.pi
            if ( x < 0 ):
                if ( y > 0):
                    m_longitude = 180 + m_longitude
                    lon = lon - m.pi
            else:
                    m_longitude = -180 + m_longitude
                    lon = m.pi + lon
        # Geocentric latitude
        latG = m.atan (z / (m.sqrt ( x * x + y * y )))
        # Geocentric latitude (of point Q, Q is intersection point of segment OP and reference ellipsoid)
        latQ = m.atan (z / ( (1 - m_e2Param ) * m.sqrt ( x * x + y * y )))
        # calculate radius of the curvature
        rCurvature = m_equatorRadius / m.sqrt (1 - m_e2Param * m.sin (latQ) * m.sin (latQ))
        # x, y, z of point Q
        xQ = rCurvature * m.cos (latQ) * m.cos (lon)
        yQ = rCurvature * m.cos (latQ) * m.sin (lon)
        zQ = rCurvature * (1 - m_e2Param) * m.sin (latQ)
        # distance OQ
        oq = m.sqrt(xQ * xQ + yQ * yQ + zQ * zQ)
        # distance PQ is OP - OQ
        pq = op - oq
        # length of the normal segment from point P of line (PO) to point T.
        # T is intersection point of linen the PO normal and ellipsoid normal from point Q.
        tp = pq * m.sin (latG - latQ)
        m_latitude =  (latQ + tp / op * m.cos (latQ - latG)) * 180.0 / m.pi
        m_altitude = pq * m.cos (latQ - latG)

    return [m_latitude, m_longitude, m_altitude]



# 限定用户收集信息区域
def limitquare():
    # SimpleScenario：以ueCenterLatLonAlt中的经纬度为中心划定用户区域
    radius = 700000  # m,小区辐射半径
    m_DayNumLen = 0.03
    EARTH_RADIUS = 6378137
    # center
    centerLatLonAlt = GeographicToCartesianCoordinates(0, 0, 0,"GRS80")
    return centerLatLonAlt, radius


# 先找可见卫星
def seesate(epoch, step):
    TLE_PATH = "./IRIDIUM-NEXT20326.txt"
    sate = satellite(75)  # 75颗卫星，但是只有66颗有效
    satlist, tle_str1, tle_str2 = sate.read_tle_file(TLE_PATH)
    position, SateId, SateBLH, undersate, beamallocate = sateposition(satlist, tle_str1, tle_str2, epoch,
                                                                      step)  # position(x,y,z),velocity,undersate(B,L,H)
    # position为卫星位置（x,y,z），SateId为卫星编号+卫星位置，undersate为卫星星下点集合，allocate为卫星坐标+小区编号+波束中心

    centerLatLonAlt, radius = limitquare()
    # print(SateId[0])
    BeamSeeAlloocate = []
    satsum = len(SateBLH)
    seesate = []
    seesateId = []
    for i in range(satsum):
        distance = m.sqrt((undersate[i][0]- centerLatLonAlt[0])**2+(undersate[i][1]- centerLatLonAlt[1])**2+(undersate[i][2]- centerLatLonAlt[2])**2)
        if (distance < 2350000-radius):
            seesate.append(SateBLH[i])
            seesateId.append(SateId[i])
            for j in range(len(beamallocate)):
                if beamallocate[j][0] == SateBLH[i]:
                    BeamSeeAlloocate.append(beamallocate[j] + [position[i]])
    # seesate为可见星的坐标集合，seesateId为卫星编号+卫星坐标BLH,BeamSeeAlloocate为卫星坐标（BLH）+小区编号+波束中心+卫星坐标（x,y,z）
    return seesate, seesateId, BeamSeeAlloocate
"""
def lonlatdistance(a, b):
    # print(a,b)
    dlon = a[0] - b[0]
    dlat = a[1] - b[1]
    a1 = m.sin(dlat / 2) ** 2 + m.cos(a[1]) * m.cos(b[1]) * m.sin(dlon / 2) ** 2
    distance = 2 * m.asin(m.sqrt(a1)) * 6371 * 1000  # 地球半径6371km
    return distance
"""

# 判断用户接入那个波束（就近原则）
def userconnectsate(ueposition, epoch, step):
    # seesate为可见星的坐标集合，seesateId为卫星编号+卫星坐标BLH,BeamSeeAlloocate为卫星坐标（BLH）+小区编号+波束中心+卫星坐标（x,y,z）
    seesatenum, SeesateId, beamallocate = seesate(epoch, step)
    uenum = len(ueposition)
    # 先找可见卫星
    satuedict = []
    Beam = []
    UeLinkSate = []
    beamnum = len(beamallocate)
    for i in range(uenum):
        ue = []
        min = []
        link = []
        beam = []
        for j in range(beamnum):
            uebeamdistance = m.sqrt((beamallocate[j][2][0]-ueposition[i][0])**2+(beamallocate[j][2][1]-ueposition[i][1])**2+(beamallocate[j][2][2]-ueposition[i][2])**2)
            if len(ue) != 0:
                mindistance = ue[0]
                if mindistance > uebeamdistance:
                    ue.remove(mindistance)
                    ue.append(uebeamdistance)
                    min.remove(min[0])
                    min.append([ueposition[i]]+ beamallocate[j])
                    link.remove(link[0])
                    link.append([ueposition[i], beamallocate[j][1]])
                    beam.remove(beam[0])
                    beam.append([beamallocate[j][1], beamallocate[j][2]])

            else:
                ue.append(uebeamdistance)
                min.append([ueposition[i]] + beamallocate[j])
                link.append([ueposition[i], beamallocate[j][1]])
                beam.append([beamallocate[j][1], beamallocate[j][2]])
        #print(min[0])

        satuedict.append(min[0])  # 用户位置+卫星位置(B,L,H)+波束编号+波束中心+卫星位置（x,y,z）
        #transfomer = Transform()
        #beam_xyz = transfomer.BLH_XYZ(beam[0][1][0], beam[0][1][1], beam[0][1][2])
        #Beam.append([beam_xyz, beam[0][0]])  # 用户位置+波束中心
        Beam.append(beam[0])
        UeLinkSate.append(link[0])  # 波束编号+波束中心
    return satuedict, Beam, UeLinkSate

