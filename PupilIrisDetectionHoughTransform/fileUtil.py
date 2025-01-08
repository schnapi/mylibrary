def FileSave(filename, content):
    with open("results/" + filename, "a+") as myfile:
        myfile.write(content)
    return True


def get_file_name(file, splitChar="/"):
    # file  = 'C:\\Users\\Sandi\\Desktop\\magistrska naloga\\IITD_database\\019\\01_L.bmp'
    temp = file.split(splitChar)
    leng = len(temp)
    filename = temp[leng - 1]
    folder = temp[leng - 2]
    database = temp[leng - 3]
    return database, folder, filename
