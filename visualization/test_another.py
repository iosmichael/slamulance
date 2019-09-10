from multiprocessing import Process
import time

def main_process():
	time.sleep(1)
	print('main process')

def another_process():
	while True:
		time.sleep(0.5)
		print('another process')

if __name__ == '__main__':
	disp = Process(target=another_process, args=())
	disp.daemon = True
	disp.start()
	while True:
		main_process()

