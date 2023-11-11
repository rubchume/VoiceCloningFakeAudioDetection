file_batches_pattern="../data/datasets/CommonVoiceFull2Try/file_batches/filelist_part_*"

for file in $file_batches_pattern; do
    echo "Extracting files from $file"
    sudo ./mount_blob_store.sh ~/blobcontainer 
    pv "../data/datasets/CommonVoiceFull2Try/common_voice_full" | sudo tar \
        -xzf - \
        --directory /home/azureuser/blobcontainer/CommonVoice/decompressed \
        -T "$file" --occurrence
    sudo umount ~/blobcontainer
    echo "Finished extracting files from $file"
done