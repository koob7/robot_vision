import numpy as np

class determine_pose:

    def find_markers(self):
        raise NotImplementedError("Implement find_markers()")

    def get_mean(self, number):
        detected_markers = {}

        for i in range(number):
            markers = self.find_markers()

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
            data["mean_tvec"] = np.mean(
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

            data["mean_R"] = R_mean

        return detected_markers