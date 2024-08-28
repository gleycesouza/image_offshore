import streamlit as st
from PIL import Image
import io
import base64
import utils  # Import the utility functions

def main():
    header_html = "<img src='data:image/png;base64,{}' class='center' style='width: 130px; display: block;margin-left: auto; margin-right: auto;'>".format(
        utils.get_base64_of_bin_file("./pictures/benthic-logo.png")
    )
    st.markdown(header_html, unsafe_allow_html=True)
    
    st.markdown('<br><h3 style="color:black; text-align: center;">Offshore Image Edit</h3>', unsafe_allow_html=True)
    utils.set_page_bg('./pictures/background-overlay2.jpg')
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file_jpg = st.file_uploader("Select JPG file to edit...", type='.jpg')
    with col2:
        uploaded_file_cr2 = st.file_uploader("Select CR2 file to rename...", type='.cr2')

    if uploaded_file_jpg is not None:
        image = Image.open(uploaded_file_jpg)
        buffer = io.BytesIO()
        image.save(buffer, 'jpeg')
        buffer.seek(0)
        bg_image = buffer

        option = st.radio('Select edit type:', ('Top', 'BB', 'Group'))

        col1, col2 = st.columns((2, 1))
        with col1:
            text_input = st.text_input("Input the image id*. Example: DGT-2152,2A1")
        with col2:
            num_image = st.text_input("Enter image number. Example: 504")

        if text_input and len(text_input)>=12:
            if option == 'Top':
                element = st.markdown('<p style="color:black; text-align: center; margin-bottom: .1em;">Editing image...</p>', unsafe_allow_html=True)
                final_img, image_edited_name = utils.process_image_top(bg_image, text_input)

            elif option == 'BB':
                element = st.markdown('<p style="color:black; text-align: center; margin-bottom: .1em;">Editing image...</p>', unsafe_allow_html=True)
                final_img, image_edited_name = utils.process_image_bb(bg_image, text_input)

            elif option == 'Group':
                num_tubes = st.number_input('Enter number of tubes:', min_value=1, max_value=4, step=1)
                tube_details = utils.get_tube_depths(num_tubes)

                # Verificar se algum item em tube_details está vazio ou é None
                if not any(detail is None or detail == "" for detail in tube_details):
                    element = st.markdown('<p style="color:black; text-align: center; margin-bottom: .1em;">Editing image...</p>', unsafe_allow_html=True)
                    final_img, image_edited_name = utils.process_image_group(num_tubes, tube_details, bg_image, text_input)
                else:
                    final_img=0
                    
            if final_img and image_edited_name:
                if uploaded_file_cr2 is None:
                    col_cr2_1, col_cr2_2 = st.columns((2, 1))
                    with col_cr2_1:
                        st.download_button('↓ Original Renamed Image.jpg', bg_image, image_edited_name + ".JPG")
                    with col_cr2_2:
                        st.download_button('↓ Edited image .jpg', final_img, image_edited_name + "_" + num_image + ".JPG")
                else:
                    if uploaded_file_jpg.name[:-4] == uploaded_file_cr2.name[:-4]:
                        col_cr2_1, col_cr2_2, col_cr2_3 = st.columns(3)
                        with col_cr2_1:
                            st.download_button('↓ Original  Renamed image.jpg', bg_image, image_edited_name + ".JPG")
                        with col_cr2_2:
                            st.download_button('↓ Original Renamed image.cr2', uploaded_file_cr2, image_edited_name + "_" + num_image + ".CR2")
                        with col_cr2_3:
                            st.download_button('↓ Edited image.jpg', final_img, image_edited_name + "_" + num_image + ".JPG")
                    else:
                        st.warning("Make sure the .cr2 and .jpg file are the same or remove the .cr2", icon="⚠️")
                
                b64img_final = base64.b64encode(final_img).decode()
                content = f'''
                    <div style="display: flex">
                        <div style="flex: 1; position: relative;border-radius: 5px;">
                            <img src='data:image/png;base64,{b64img_final}' class='center' style='padding-right:10px;display: block; margin: auto; max-width: 40%'>
                        </div>
                    </div>'''
                st.markdown(f'<p style="color:black; text-align: center; margin-bottom: .1em;">Edited image preview:</p>', unsafe_allow_html=True)
                st.markdown(content, unsafe_allow_html=True)
                st.markdown(f'<p style="color:black; text-align: center; margin-bottom: .1em;">{image_edited_name + "_" + num_image + ".JPG"}</p>', unsafe_allow_html=True)
                element.empty()
            else:
                st.warning("Please fill out all tube depth fields before continuing.")

if __name__ == "__main__":
    main()
