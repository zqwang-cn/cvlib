import subprocess


class RTMPSender:
    def __init__(self, width, height, fps, url):
        command = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', "{}x{}".format(width, height),
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv', 
                url]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def send(self, img):
        self.process.stdin.write(img.tostring())

    def stop(self):
        self.process.kill()