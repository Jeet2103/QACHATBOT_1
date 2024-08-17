import streamlit as st

# Initialize a variable to store the number, using Streamlit's session state
if 'number' not in st.session_state:
    st.session_state.number = 0

# Define a function to increase the number
def increase_number():
    st.session_state.number += 1

# Create a button that, when pressed, calls the function to increase the number
if st.button('Increase Number'):
    increase_number()

# Display the current number
st.write(f'Current Number: {st.session_state.number}')
