import streamlit as st

st.title("🚀 Minitorch Streamlit App")
st.write("This is a test deployment for Streamlit.")

# Check if minitorch is working
try:
    import minitorch.operators as operators
    st.success("✅ Minitorch imported successfully!")
except Exception as e:
    st.error(f"❌ Error importing minitorch: {e}")

