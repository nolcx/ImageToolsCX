from rembg import remove
from rembg import new_session
from PIL import Image

input_path = 'Profile.png'
output_path = 'FinalBGRem.png'

input = Image.open(input_path)

# Name model
#nameModel = 'isnet-general-use'
nameModel = 'birefnet-portrait'

# Session initialization 
# (initialization consume a lot of resources, so if you want to transform 
# more than one image, you should considere to use a loop)
rembg_session = new_session(nameModel)

# If you want to use a specific model, 
# you need to send the session in the remove call
output = remove(input, session=rembg_session)
#output = remove(input)

# Save the image in the output path
output.save(output_path)