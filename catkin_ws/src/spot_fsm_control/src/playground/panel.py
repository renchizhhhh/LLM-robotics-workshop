import tkinter as tk
from main import SpotControlInterface
from PIL import ImageTk, Image
import cv2


class SpotControlPanel(SpotControlInterface):

    def __init__(self):
        super(SpotControlPanel, self).__init__()

    def load_panel(self):
        m = tk.Tk()
        m.title("Spot control panel")
        # m.geometry("300x150+1000+1000")

        self.arm_control_frame(m)
        self.movement_control_frame(m)
        self.video_stream_frame(m)

        m.mainloop()

    def video_stream_frame(self, m, column=2):
        l = tk.Label(m, text = "Camera")
        l.config(font =("Courier", 18))
        l.grid(column=column, row=0)

        app = tk.Frame(m, bg="white")
        app.grid(column=column)
        # Create a label in the frame
        lmain = tk.Label(app)
        lmain.grid()

        cap = cv2.VideoCapture(0)


        def video_stream():
            # frame = self.image_client.get_image_from_sources(["hand_color_image"])
            _, frame = cap.read()
            frame = cv2.resize(frame, (400, 300))
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            lmain.after(1, video_stream) 

        video_stream()

    def move_forward(self):
        self.forward, self.strafe, self.rotate = 0, 0, 0
        self.forward = 0.4
        self.move_command()
        
    def move_backward(self):
        self.forward, self.strafe, self.rotate = 0, 0, 0
        self.forward = -0.4
        self.move_command()

    def move_left(self):
        self.forward, self.strafe, self.rotate = 0, 0, 0
        self.strafe = 0.4
        self.move_command()

    def move_right(self):
        self.forward, self.strafe, self.rotate = 0, 0, 0
        self.strafe = -0.4
        self.move_command()

    def stand_high(self):
        self.stand(0.5)

    def movement_control_frame(self, m, column=0):
        l = tk.Label(m, text = "Movement control", width=30)
        l.config(font =("Courier", 18))
        l.grid(column=column, row=0)

        redbutton = tk.Button(m, text = 'Keyboard control', command=self.keyboard_movement_control)
        redbutton.grid(column=column, row=1)
        greenbutton = tk.Button(m, text = 'Stand high', command=self.stand_high)
        greenbutton.grid(column=column, row=2)
        bluebutton = tk.Button(m, text ="Sit down", command=self.sit_down)
        bluebutton.grid(column=column, row=3)

        frame_control = tk.Frame(m)
        frame_control.grid(column=column, row=4)

        bluebutton = tk.Button(frame_control, text ="A", command=self.move_forward)
        bluebutton.grid(column=1, row=0)

        bluebutton = tk.Button(frame_control, text ="<", command=self.move_left)
        bluebutton.grid(column=0, row=1)

        bluebutton = tk.Button(frame_control, text ="O", command=self.stop)
        bluebutton.grid(column=1, row=1)

        bluebutton = tk.Button(frame_control, text =">", command=self.move_right)
        bluebutton.grid(column=2, row=1)

        bluebutton = tk.Button(frame_control, text ="v", command=self.move_backward)
        bluebutton.grid(column=1, row=2)

    def arm_control_frame(self, m, column=1):
        l = tk.Label(m, text = "Arm Control", width=30)
        l.config(font =("Courier", 18))
        l.grid(column=column, row=0)

        w = tk.Button(m, text="Manual control", command=self.manual_manipulator_control)
        w.grid(column=column, row=1)
        w = tk.Button(m, text="arm gaze", command=self.gaze_control)
        w.grid(column=column, row=2)
        w = tk.Button(m, text="Pickup and object", command=self.arm_object_grasp)
        w.grid(column=column, row=3)
        w = tk.Button(m, text="Walk to object", command=self.walk_to_object)
        w.grid(column=column, row=4)
        

if __name__ == "__main__":
    panel = SpotControlPanel()
    panel.load_panel()