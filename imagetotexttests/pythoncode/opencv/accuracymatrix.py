import pandas as pd

df = pd.read_csv("Sample Resources/pdfverifier.csv")
filename = "P6_2019_English_SA1_Catholic_High"
print(df)
print("\n")
verifier = df.set_index("Paper Name", drop=False)
if any(verifier["Paper Name"].values == filename):
    num_qns = int(verifier.loc[filename, "Questions"])
    num_images = int(verifier.loc[filename, "Images"])
    detectedimagecount = 10
    detectedqncount = 70
    qn_acc = (detectedqncount/num_qns) * 100
    img_acc = (detectedimagecount/num_images) * 100
    print("Accuracy of Question Numbers: " + str(qn_acc) + "%")
    print("Accuracy of Images : " + str(img_acc) + "%")
else:
    pass