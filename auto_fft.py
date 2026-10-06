import argparse
import os
import sys

# game plan:
# get file from URL
# strictly requires two arguments: url and filename
# can specify extra arguments for waveform colors
# run file_transfrom.py

if __name__ == "__main__":
    os.system("clear")
    print(sys.argv)
    print(len(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--name")
    parser.add_argument("--single-color", default="white")
    parser.add_argument("--colors", default=["none", "none", "none", "none"], nargs=4)
    parser.add_argument("--topmax", default="none")
    parser.add_argument("--topwave", default="none")
    parser.add_argument("--bottomwave", default="none")
    parser.add_argument("--bottommin", default="none")

    args = parser.parse_args()

    topmax = args.single_color
    topwave = args.single_color
    bottomwave = args.single_color
    bottommin = args.single_color

    if args.colors[0] != "none":
        topmax = args.colors[0]
    if args.colors[1] != "none":
        topwave = args.colors[1]
    if args.colors[2] != "none":
        bottomwave = args.colors[2]
    if args.colors[3] != "none":
        bottommin = args.colors[3]

    if args.topmax != "none":
        topmax = args.topmax
    if args.topwave != "none":
        topwave = args.topwave
    if args.bottomwave != "none":
        bottomwave = args.bottomwave
    if args.bottommin != "none":
        bottommin = args.bottommin

    url = args.url
    filename = args.name

    os.system(f'yt-dlp -x --audio-format wav -o "{filename}.%(ext)s" "{url}"')
    if len(sys.argv) > 3:
        os.system(
            f'python3 file_transform.py {filename}.wav "{topmax}" "{topwave}" "{bottomwave}" "{bottommin}"'
        )
    else:
        os.system(f"python3 file_transform.py {filename}.wav")

    os.system(f"rm {filename}.wav")
    os.system(f"open {filename}_animated.mp4")
