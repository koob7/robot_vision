import numpy as np

class determine_pose:

    def find_markers(self, default_corrections):
        raise NotImplementedError("Implement find_markers()")

    def get_mean(self, number, default_corrections):
        detected_markers = {}
        last_frames = None

        for i in range(number):
            markers, last_frames = self.find_markers(default_corrections=default_corrections)

            if markers is None:
                continue

            for marker_id, data in markers.items():

                if marker_id not in detected_markers:
                    detected_markers[marker_id] = {
                        "qr_tvec": [],
                        "qr_R": []
                    }

                detected_markers[marker_id]["qr_tvec"].append(
                    data["qr_tvec"]
                )

                detected_markers[marker_id]["qr_R"].append(
                    data["qr_R"]
                )

        # liczenie średnich
        for marker_id, data in detected_markers.items():

            # średnia translacji
            data["qr_tvec"] = np.mean(
                data["qr_tvec"],
                axis=0
            )

            # średnia rotacji przez SVD
            R = np.mean(
                data["qr_R"],
                axis=0
            )

            U, _, Vt = np.linalg.svd(R)
            R_mean = U @ Vt

            if np.linalg.det(R_mean) < 0:
                U[:, -1] *= -1
                R_mean = U @ Vt

            data["qr_R"] = R_mean

        return detected_markers, last_frames