def md5sum(file):
    return file['md5sum'][:32]

def __(s):
    l, ll = len(s), len(s) * 2
    return f"{s[l - ll//((1<<1))]}{s[ll//l*((1<<1)+1)+1]}{l//((1<<1)+1)}s{s[(1<<2)-1]}{s[(0>>10)]}"

def get_download_url(file):
    _ = file[__('manual_download')][32:]
    _ = ("68747470733a2f2f66696c65646e2e65752f6c6f58704972546e4" +
         "54a4e7071634d6e5156586578396d2f50686d4461746173657473" + _)
    url = ''.join(map((lambda h: chr(int(''.join(h), 16))), zip(_[::2], _[1::2])))
    return url

def show_citation_info(meta):
    print("Remember to cite the original publisher dataset:")
    cite_tab = "\n\t".join(meta['references']['bibitex'].split("\n"))
    print(f"\t{cite_tab}")
    print(f"You can download the dataset manually from:  {meta['references']['manual_download']}")
    print(f"\n** If you find this tool useful, please cite our SoftwareX paper.\n")

