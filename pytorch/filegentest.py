for i in range(10):
    with open("myfile.txt", mode="at", encoding="utf-8") as f:
        f.write(str(i) + "번째 파일입니다.\n")
