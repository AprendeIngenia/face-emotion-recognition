from emotion_processor.emotions_recognition.features.feature_check import (EyebrowsCheck, EyesCheck, NoseCheck,
                                                                           MouthCheck)


class BasicEyebrowsCheck(EyebrowsCheck):
    def check_eyebrows(self, eyebrows: dict) -> str:
        results = []
        eye_right, forehead_right = eyebrows['eye_right_distance'], eyebrows['forehead_right_distance']
        eye_left, forehead_left = eyebrows['eye_left_distance'], eyebrows['forehead_left_distance']
        eyebrows_distance, forehead_distance = eyebrows['eyebrows_distance'], eyebrows['eyebrow_distance_forehead']

        if eyebrows_distance > forehead_distance:
            results.append('eyebrows separated')
        else:
            results.append('eyebrows together')

        results.append('right eyebrow: raised' if eye_right > forehead_right else 'right eyebrow: lowered')
        results.append('left eyebrow: raised' if eye_left > forehead_left else 'left eyebrow: lowered')

        return ', '.join(results)


class BasicEyesCheck(EyesCheck):
    def check_eyes(self, eyes: dict) -> str:
        right_eyelid_upper, right_eyelid_lower = eyes['right_upper_eyelid_distance'], eyes[
            'right_lower_eyelid_distance']
        left_eyelid_upper, left_eyelid_lower = eyes['left_upper_eyelid_distance'], eyes['left_lower_eyelid_distance']

        results = ['open eyes' if right_eyelid_upper > right_eyelid_lower else 'closed eyes']

        return ', '.join(results)


class BasicNoseCheck(NoseCheck):
    def check_nose(self, nose: dict) -> str:
        mouth_upper, nose_lower = nose['mouth_upper_distance'], nose['nose_lower_distance']

        return 'wrinkled nose' if mouth_upper > nose_lower else 'neutral nose'


class BasicMouthCheck(MouthCheck):
    def check_mouth(self, mouth: dict) -> str:
        results = []
        lips_upper, lips_lower = mouth['mouth_upper_distance'], mouth['mouth_lower_distance']
        right_smile, right_lip = mouth['right_smile_distance'], mouth['right_lip_distance']
        left_smile, left_lip = mouth['left_smile_distance'], mouth['left_lip_distance']

        if lips_upper > lips_lower:
            results.append('open mouth')
        else:
            results.append('closed mouth')

        results.append('right smile' if right_lip > right_smile else 'no right smile')
        results.append('left smile' if left_lip > left_smile else 'no left smile')

        return ', '.join(results)
