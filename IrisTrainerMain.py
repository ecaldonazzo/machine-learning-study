from src.projects.IrisProject import IrisProject


class IrisTrainerMain:
    @staticmethod
    def run():
        project = IrisProject()
        project.run()

if __name__ == "__main__":
    IrisTrainerMain.run()
