import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle


st.image('./pic/wel.jpg')

html_8="""
<div 
            style="background-color:#fffacd;
            padding:5px;
            border-radius:0px 0px 0px 0px;
            border-style:'solid';
            border-color:white">
<center><h3>ระบบการตัดสินใจเลือกซื้อเครื่องคอมพิวเตอร์โน้ตบุ๊ค ด้วยเทนนิค KNN</h3></center>
</div>
"""

st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

dt=pd.read_csv("./data/Smote.csv")
st.write(dt.head(9))
dt1 = dt['age'].sum()
dt2 = dt['objective'].sum()
dt3 = dt['price'].sum()
dt4 = dt['store'].sum()
dt5 = dt['motivation'].sum()
dt6 = dt['parent_income'].sum()
dx=[dt1,dt2,dt3,dt4,dt5,dt6]
dx2=pd.DataFrame(dx,index=["d1","d2","d3","d4","d5","d6"])


html_8="""
<div style="background-color:#fffacd;
            padding:5px;
            border-radius:0px 0px 0px 0px;
            border-style:'solid';
            border-color:white">
<center><h3>กรอกข้อมูลการตัดสินใจเลือกซื้อ</h3></center>
</div>
"""

st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

age=st.radio("อายุ:1 <20, 2 20-25 , 3 >25 ",(1,2,3))
objective=st.radio("วัตถุประสงค์ในการเลือกซื้อคอมพิวเตอร์โน้ตบุ๊ค:1.เพื่อการศึกษา2.เพื่อใช้ติดต่อสื่อสาร3.เพื่อสร้างรายได้เสริม4.เพื่อเล่นเกม",(1,2,3,4,5))
price=st.radio("ราคาเครื่องคอมพิวเตอร์โน้ตบุ๊ค:1. <15000 2. 15001-20000 3. 20001-25000 4. 25001-30000 5. >30001",(1,2,3,4,5))
store=st.radio("ท่านเลือกซื้อคอมพิวเตอร์โน้ตบุ๊คจากที่ใด: 1. ร้านค้าคอมพิวเตอร์ทั่วไป 2. ร้านคอมพิวเตอร์ในศูนย์การค้า 3. ในห้างสรรพสินค้า 4. ร้านตัวแทนที่ได้รับการแต่งตั้งให้จำหน่ายยี่ห้ออย่างเป็นทางการ",(1,2,3,4))
motivation=st.radio("มูลเหตุจูงใจหรือบุคคลใดที่มีอิทธิพลต่อกำรตัดสินใจเลือกซื้อคอมพิวเตอร์โน้ตบุ๊คมากที่สุด : 1.บุคคลในครอบครัว 2. การโฆษณา 3. เพื่อน/ญาติ พี่น้อง 4. ตัดสินใจซื้อด้วยตัวเอง",(1,2,3,4))
parent_income=st.radio("รายได้ของผู้ปกครอง: 1 ต่ำกว่า 15000 บาท ,2 15001-20000, 3 20001-25000, 4 25001-30000, 5 30000+",(1,2,3,4,5))

if st.button("ทำนายผล"):
    loaded_model = pickle.load(open('./data/smote_model.sav', 'rb'))
    input_data =  (age,objective,price,store,motivation,parent_income)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    st.write(prediction)
    if prediction == 'Acer':
        st.image('./pic/AC.jpg')
    elif prediction == 'Asus':
        st.image('./pic/AS.jpg')
    elif prediction == 'Dell':
        st.image('./pic/D.jpg')
    elif prediction == 'HP':
        st.image('./pic/H.jpg')
    elif prediction == 'Lenovo':
        st.image('./pic/L.jpg')
    else:
        st.image('./pic/M.jpg')
else:
    st.write("")


