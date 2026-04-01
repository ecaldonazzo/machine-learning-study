from src.projects.NaiveBayesIrisProject import NaiveBayesIrisProject


class NaiveBayesIrisMain:
    @staticmethod
    def run():
        project = NaiveBayesIrisProject()
        project.run()

if __name__ == "__main__":
    NaiveBayesIrisMain.run()