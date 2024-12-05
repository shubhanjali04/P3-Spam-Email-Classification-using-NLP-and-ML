import streamlit as st
import pickle

model = pickle.load(open('spam.pkl','rb'))
cv = pickle.load(open('vec.pkl','rb'))

def main():
	st.title("Email Spam Classification App")
	st.write("Machine Learning app to detect emails as spam or Not spam.")
	st.subheader("Classification")
	user_input=st.text_area("Enter your Message" ,height=150)
	if st.button("Classify"):
		if user_input:
			data=[user_input]
			print(data)
			vec=cv.transform(data).toarray()
			result=model.predict(vec)
			if result[0]==0:
				st.success("This is Not A Spam Email")
			else:
				st.error("Spam Email")
		else:
			st.write("Please enter an email to classify.")
main()