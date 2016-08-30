import os

l = ["Head", "L_Ankle", "L_Elbow", "L_Hip", "L_Knee", "L_Shoulder", "L_Toes", "L_Wrist", "R_Ankle", "R_Elbow", "R_Hip", "R_Knee", "R_Shoulder", "R_Toes", "R_Wrist"]

for word in l:
    os.system("convert -fill black -background white -bordercolor white -border 4 -font futura-normal -pointsize 18 label:\"%s\" \"%s.png\""%(word, word))
