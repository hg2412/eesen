import sys

train_scps = ["train_local.scp", "train_tmp.scp"]
cv_scps = ["cv_local.scp", "cv_tmp.scp"]

tr_ark = raw_input("Train ark file?\n")
cv_ark = raw_input("Cross validation ark file?\n")
scp_dir = raw_input("Scp dir?\n")

print("Preprocessing scp data:")
for scp in train_scps:
    process_scp(config["scp_dir"] + "/" + scp, config["data_dir"] + "/" + scp, "/home/haoxiang/Desktop/eesen-tf_clean/data/am_data/train.ark", config["tr_ark"])

for scp in cv_scps:
    process_scp(config["scp_dir"] + "/" + scp, config["data_dir"] + "/" + scp, "/home/haoxiang/Desktop/eesen-tf_clean/data/am_data/cv.ark", config["cv_ark"])
