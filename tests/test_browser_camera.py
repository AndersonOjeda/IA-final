"""Browser transport: validated frames, one result, acknowledgments and lifecycle."""
import base64
import io
import unittest

import numpy as np
from PIL import Image

from app.browser_camera import BrowserRecognition
from app.detector import Detection
from app.localization import Region
from tests.test_app import FakeService


def payload(sid='session-a',sequence=1):
    image=io.BytesIO();Image.new('RGB',(160,120)).save(image,'JPEG')
    return {'event':'frame','stream_id':sid,'sequence':sequence,'image':base64.b64encode(image.getvalue()).decode()}


class Detector:
    calls=0
    def detect(self,frame):
        self.calls+=1
        return [Detection((10,20,50,90),.9,Region((5,15,55,95),'neuronal')),
                Detection((100,20,150,100),.7,Region((95,15,155,105),'neuronal'))]


class BrowserCameraTests(unittest.TestCase):
    def test_frame_reaches_own_models_once_and_box_is_normalized(self):
        recognition=BrowserRecognition();detector=Detector();classifier=FakeService();frame=payload()
        self.assertTrue(recognition.consume(frame,detector,classifier))
        self.assertEqual(detector.calls,1);self.assertEqual(classifier.calls,1)
        self.assertEqual(len(recognition.objects),1)
        np.testing.assert_allclose(recognition.response['detection']['box'],[10/160,20/120,50/160,90/120])
        self.assertEqual(recognition.response['sequence'],1)
        self.assertFalse(recognition.consume(frame,detector,classifier))
        self.assertEqual(detector.calls,1)

    def test_stop_clears_result_and_late_frames_do_not_restart_old_stream(self):
        recognition=BrowserRecognition();detector=Detector();classifier=FakeService()
        recognition.consume(payload(),detector,classifier)
        recognition.consume({'event':'stopped','stream_id':'session-a','sequence':2},detector,classifier)
        self.assertFalse(recognition.running);self.assertEqual(recognition.objects,())
        self.assertFalse(recognition.consume(payload(sequence=1),detector,classifier))
        recognition.consume(payload('session-b'),detector,classifier)
        self.assertTrue(recognition.running)
        self.assertFalse(recognition.consume(payload('session-a',3),detector,classifier))
        self.assertEqual(detector.calls,2)

    def test_invalid_frame_is_acknowledged_without_running_models(self):
        for image in ['not_base64','x'*2_000_001]:
            recognition=BrowserRecognition();detector=Detector();classifier=FakeService()
            frame=payload();frame['image']=image
            recognition.consume(frame,detector,classifier)
            self.assertIn('error',recognition.response)
            self.assertEqual(recognition.response['sequence'],1)
            self.assertEqual(detector.calls,0);self.assertEqual(classifier.calls,0)
            self.assertEqual(recognition.objects,())


if __name__=='__main__':unittest.main()
