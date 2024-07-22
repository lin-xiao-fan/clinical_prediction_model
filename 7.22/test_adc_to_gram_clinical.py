import csv
import sys
import os
import pandas as pd

#clincal

HALF_FEATURE = 6
ALL_FEATURE = 12
TIME_WINDOW = 1000



def adc_to_v( adc ) :
    v = 0
    v = adc / 4095 * 3.3
    #print("adc : ", adc , " v : " , v ) 
    return v



def v_to_gram( subject , foot , v , state ) : # gram = a * v**3 + b * v**2 + c * v + d 
    gram = 0 
    #print("subject : ", type ( subject ) , " foot : " , foot, " v : ", v, " state : ", type( state )  ) 
    if subject == 1 : # 男
        if foot == "left" : # 左腳
            if state == " ha" : #  [ 1678.1 , -1960.9 , 1833.3 , -7.2303 ]
                gram = 1678.1 * v**3 + -1960.9 * v**2 + 1833.3 * v + -7.2303 
            elif state == " lt" : #  [ 2006.9 , -1718.5 , 2123.6 , -0.5857 ]
                gram = 2006.9 * v**3 + -1718.5  * v**2 + 2123.6 * v + -0.5857
            elif state == " m1" : #  [ 2135.1 , -3218.9 , 3507.6 , 10.45 ]
                gram = 2135.1 * v**3 + -3218.9 * v**2 + 3507.6 * v + 10.45 
            elif state == " m5" :  # [ 2653 , -3881.4 , 3254.1 , 0.3377 ]
                gram = 2653 * v**3 + -3881.4  * v**2 + 3254.1 * v + 0.3377  
            elif state == " arch" : # [ 2895.1 , -2745.5 , 2881.9 , -17.436 ]
                gram = 2895.1 * v**3 + -2745.5  * v**2 + 2881.9 * v + -17.436
            elif state == " mh" : # [ 1617.8 , -2194.3 , 2342.8 , -8.9142 ]
                gram = 1617.8 * v**3 + -2194.3  * v**2 + 2342.8 * v + -8.9142 
        elif foot == "right" : #右腳
            if state == " ha" : # [ 1364.4 , -1468.4 , 2312.4 , -17.676 ]
                gram = 1364.4 * v**3 + -1468.4 * v**2 + 2312.4 * v + -17.676 
            elif state == " lt" : # [ 2056.2 , -3043.4 , 2740.9 , 2.8214 ]
                gram = 2056.2 * v**3 + -3043.4 * v**2 + 2740.9 * v + 2.8214   
            elif state == " m1" : # [ 1438.8 , -1765.4 , 2387.9 , -2.581 ]
                gram = 1438.8 * v**3 + -1765.4 * v**2 + 2387.9 * v + -2.581
            elif state == " m5" : # [ 2084.6 , -2933 , 2884.5 , -6.5823 ]
                gram = 2084.6 * v**3 + -2933 * v**2 + 2884.5 * v + -6.5823
            elif state == " arch" : # [ -395.32 , 1249.1 , 1798.3 , -14.459 ]
                gram = -395.32 * v**3 + 1249.1 * v**2 + 1798.3 * v + -14.459
            elif state == " mh" : # [ 180.73 , -385.13 , 2646.4 , -1.6599 ]
                gram = 180.73 * v**3 + -385.13 * v**2 + 2646.4 * v + -1.6599
    elif subject in ( 2 , 3 , 4 , 5 , 6 ) : #女
        if foot == "left" : # 左腳
            if state == " ha" : # [ 2230.2 , -2417.6 , 2613.4 , 9.4971 ]
                gram = 2230.2 * v**3 + -2417.6 * v**2 + 2613.4 * v + 9.4971
            elif state == " lt" : # [ 1210.7 , -2009.1 , 2376.8 , -13.679 ]
                gram = 1210.7 * v**3 + -2009.1 * v**2 + 2376.8 * v + -13.679
            elif state == " m1" : # [ 1647.2 , -2513.9 , 2568.8 , 7.823 ]
                gram =  1647.2 * v**3 + -2513.9 * v**2 + 2568.8 * v + 7.823 
            elif state == " m5" : # [ 2803.1 , -4056 , 3193.1 , 23.825 ]
                gram = 2803.1 * v**3 + -4056 * v**2 + 3193.1 * v + 23.825
            elif state == " arch" : # [ 1060.9 , -1073.7 , 2205.8 , -11.123 ]
                gram = 1060.9 * v**3 + -1073.7 * v**2 + 2205.8 * v + -11.123
            elif state == " mh" : # [ 2399 , -3047.7 , 2861.7 , 10.982 ]
                gram = 2399 * v**3 + -3047.7 * v**2 + 2861.7 * v + 10.982
        elif foot == "right" : #右腳
            if state == " ha" : # [ 435.82 , -362.6 , 1655.8 , -11.403 ]
                gram = 435.82 * v**3 + -362.6 * v**2 + 1655.8 * v + -11.403
            elif state == " lt" : # [ -51.856 , 654.91 , 869.7 , 15.259 ]
                gram = -51.856 * v**3 + 654.91 * v**2 + 869.7 * v + 15.259  
            elif state == " m1" : # [ 906.59 , -1215.8 , 2426.1 , -48.587 ]
                gram = 906.59 * v**3 + -1215.8 * v**2 + 2426.1 * v + -48.587 
            elif state == " m5" : # [ 1199.5 , -1267.2 , 2044.1 , 21.001 ]
                gram = 1199.5 * v**3 + -1267.2 * v**2 + 2044.1 * v + 21.001
            elif state == " arch" : # [ 1480.3 , -2059.9 , 2955.5 , -12.366 ]
                gram = 1480.3 * v**3 + -2059.9 * v**2 + 2955.5 * v + -12.366
            elif state == " mh" : # [ 2678.8 , -3825.8 , 3536.8 , -27.348 ]
                gram = 2678.8 * v**3 + -3825.8 * v**2 + 3536.8 * v + -27.348

    #print ( " gram : " , gram )
    return gram








def read_csv( filename, subject , foot ):
    data = []

    #print ( subject , foot )
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        head = next(reader, None) #跳過第一行參數名稱
        print( subject ,len(head) )

            

        #test = next(reader, None)
        #print( filename )
        j = 0 
        for row in reader :
            j = j + 1
            timestamp = row[1]  # second column
            #在這裡處理adc轉gram
            start_index = 3
            # 資料校正區
            #print(subject,foot,row)
            for i in range( start_index, 9 ) : #每行最後6個進行校正       
            #    if row[i] == '' :
            #        row[i] = 0.0
            #    print("subject : ", subject , " foot : " , foot,  " row : " , test[i] ) 



                try:
                   timestamp_int = int(timestamp)
                except ValueError:
                    continue

               
                row[i] = float( v_to_gram( subject , foot , adc_to_v( float ( row[i] ) ) , head[i]  ) )
            #    print("subject : ", subject , " foot : " , foot,  " afterrow : " , test[i] ) 

            
            data.append([float(i) for i in row[3:9]] + [timestamp_int])




    return data


def extract_max(left_data, right_data):
    peaks = [-1] * ALL_FEATURE

    left_count = 1
    right_count = 1
    current_timestamp = max(left_data[0][HALF_FEATURE],
                            right_data[0][HALF_FEATURE])
    result = []
    while left_count < len(left_data) or right_count < len(right_data):
        if left_count == len(left_data):
            is_left = False
        elif right_count == len(right_data):
            is_left = True
        else:
            # compare timestamp
            is_left = (left_data[left_count][HALF_FEATURE] <=
                       right_data[right_count][HALF_FEATURE])

        if is_left:
            row = left_data[left_count]
            left_count += 1
        else:
            row = right_data[right_count]
            right_count += 1

        if row[HALF_FEATURE] > current_timestamp + TIME_WINDOW and -1 not in peaks:
            result.append("%s\n" % ",".join("%.2f" % i for i in peaks))
            peaks = [-1] * ALL_FEATURE
            current_timestamp = row[HALF_FEATURE]
            
        offset = 0 if is_left else HALF_FEATURE
        for i in range(HALF_FEATURE):
            peaks[i + offset] = max(peaks[i + offset], row[i])

        
    return result


def main():
    if len(sys.argv) < 2:
        sys.exit("parse_input.py input_dir")

    input_dir = sys.argv[1]
    for k in range(1, 7):
        for runs in ["pre","post"]:
        

            left_data = read_csv(os.path.join(
                input_dir, "Subject_clinical_%d_%s_left_raw.csv" % (k, runs)), k , "left" )
            right_data = read_csv(os.path.join(
                input_dir, "Subject_clinical_%d_%s_right_raw.csv" % (k, runs)), k , "right" )

            
            result = extract_max(left_data, right_data)
            #result.insert( 0,["ha","lt","m1","m5","arch","mh","ha","lt","m1","m5","arch","mh"])

            with open("clinical_gram/test_clinical_%d_%s_gram.csv" % (k, runs), "w") as f:
                
                head = "ha,lt,m1,m5,arch,mh,ha,lt,m1,m5,arch,mh\n"
                f.write(head)
                f.writelines(result[1:-1])


if __name__ == "__main__":
    main()

