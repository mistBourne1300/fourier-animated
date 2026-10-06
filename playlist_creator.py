import os
import time

PLAYLIST_FOLDER = "playlist_files"
PLAYLIST_DESCRIPTOR_FILE = "playlist.txt"

if __name__ == "__main__":
    num_created = 0
    start = time.time()
    with open(
        os.path.join(os.getcwd(), PLAYLIST_FOLDER, PLAYLIST_DESCRIPTOR_FILE)
    ) as file:
        for i, line in enumerate(file.readlines()):
            command = "python3 auto_fft.py --url"
            if len(line.strip().split()) == 0:
                continue
            if line[0] == "#":
                print(f"skipping line {i}")
                continue
            filename = line.strip().split()[1]

            for i, arg in enumerate(line.strip().split()):
                command = command + f' "{arg}"'
                if i == 0:
                    command = command + f" --name"
                elif i == 1:
                    command = command + f" --colors"

            try:
                os.system(command)
                os.rename(
                    os.path.join(os.getcwd(), f"{filename}_animated.mp4"),
                    os.path.join(os.getcwd(), PLAYLIST_FOLDER, f"{filename}.mp4"),
                )
                num_created += 1
            except Exception as e:
                if type(e) == KeyboardInterrupt:
                    break
                with open(
                    os.path.join(os.getcwd(), PLAYLIST_FOLDER, "failed_videos.txt"), "a"
                ) as failed_file:
                    failed_file.write(f"{filename} failed. exception: {e}\n\n\n\n\n")
    computation_time = time.time() - start
    print(f"{num_created} videos took {computation_time:.5f} sec.")
    print(f"that's {computation_time/num_created:.5f} sec/video")
