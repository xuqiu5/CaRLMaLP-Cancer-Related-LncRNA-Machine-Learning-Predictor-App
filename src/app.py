from typing import List

from shiny import *
from shiny.types import NavSetArg

from ml_tools import run_ML_pipeline, run_PCA
import pandas as pd

def nav_controls() -> List[NavSetArg]:
    """Navigation tabs at the top of Shiny app"""
    return (
        ui.nav(

            "Machine Learning",  # The title of the the current navigation bar

            ui.panel_title('Run machine learning model training'), 

            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.input_file("features", "Upload the feature set (should be .csv/.tsv file, the first column corresponds to target set)", accept=[".csv", '.tsv'], multiple=False),
                    # ui.input_file("user_input", "Upload the input, should have the exact same format as the feature set (should be .csv/.tsv file, the first column corresponds to target set)", accept=[".csv", '.tsv'], multiple=False),
                    # ui.input_file("pos_target", "Upload the positive target set (should be a list of sample IDs formatted as a column)", multiple=False),
                    # ui.input_file("neg_target", "Upload the negative target set (should be a list of sample IDs formatted as a column)", multiple=False),
                    ui.input_select(
                        'ml_model', 
                        'Choose ML model to train', {
                            'NB': 'Naive Bayes Classifier',
                            'kNN': 'kNN classifier',
                            'LR': 'Logistic Regression',
                            'RF': 'Random Forest',
                            'SVM': 'SVM classifier'
                            }
                    ),
                    ui.input_action_button(
                    "run", "Run training"
                )
                ),
                ui.panel_main(
                    ui.navset_tab(
                        # ui.nav(
                        #     "Classification metrics", 
                        #     ui.output_table(
                        #         'classification_metrics', 
                        #         )
                        # ),
                        ui.nav(
                            "Prediction Result",
                            ui.output_table(
                                'prediction'
                                )
                        ),
                        ui.nav(
                            "Confusion matrix", 
                            ui.output_plot(
                                'confusion_matrix', 
                                width='500px', 
                                height='500px'
                                )
                        ),
                        ui.nav(
                            "ROC AUC curve", 
                            ui.output_plot(
                                'roc_auc_curve', 
                                width='500px', 
                                height='500px'
                                )
                        ),
                        ui.nav(
                            'Feature importance',
                            ui.output_plot(
                                'feature_importance',
                                width='1000px', 
                                height='500px'
                            )
                        ),
                    )
                )
            )
        ),
        ui.nav(
            "PCA",  
            ui.panel_title('Principal Component Analysis'),

            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.input_action_button(
                    "run_pca", "Run analysis"
                    )
                ),
                ui.panel_main(
                    ui.div(
                        {"class": "card mb-3"},
                        ui.div(
                            {"class": "card-body"},
                            ui.h5({"class": "card-title mt-0"}, "Principal components"),
                            ui.output_plot("pca", width='800px', height='500px')
                        )
                    )
                )
            )
        ),
        ui.nav(
            "How It Works", 
             ui.h1("""How It Works"""),
             ui.p("""Cancer-Related LncRNA Machine Learning Predictor App (CaRLMaLP R-Shiny App) is a Web-based user interface for predicting features that play key roles in classifying cancer-related lncRNAs by integrating manifold features (genomic, expression, epigenetic and network features) with five machine-learning techniques (Random Forest (RF), Naïve bayes (NB), Support Vector Machine (SVM), Logistic Regression (LR) and K-Nearest Neighbors (KNN). With this App, predicting novel cancer-related lncRNA with high accuracy and efficiency is possible. This App was developed using R Shiny. After preprocessing your integrated features (example can be found here https://github.com/xuqiu5/CaRLMaLP-Cancer-Related-LncRNA-Machine-Learning-Predictor-App/blob/main/README.md, a feature_set.csv file, positive.lncRNA.glist.xls and negative.lncRNA.glist.xls files are required to predict features that play key roles in classifying cancer-related lncRNAs.  
             """),
             ui.h2("""Result Interpretation 
             """),
             ui.h3("""Machine Learning Panel
             """),
             ui.p("""After uploading features for each Gene_ID, users can choose a machine learning algorithm from the drop-down menu. Then click run to start the machine learning. Since the user input may have less number of features, this app provides the confusion matrix and AUC ROC curve as indicators for the robustness.
             """),
             ui.h4("""Prediction Result
             """),
             ui.p("""The prediction result contains a table: the first column is the Gene_ID provided by the user, the second column is the prediction result. One (1) means positive cancer-related, and -1 means negative cancer-unrelated.
             """),
             ui.h4("""Confusion Matrix
             """),
             ui.p("""The confusion matrix plot counts the number of true positives, true negatives, false positives, and false negatives for the trained model. The higher the number of true positives and true negatives, the more robust the model is.

             """),
             ui.h4("""AUC ROC Curve
             """),
             ui.p("""Area under the ROC curve demonstrates the accuracy of the model trained, also indicating the accuracy of the prediction. The closer the area under curve to 1, the higher the accuracy of the prediction.
             """),
             ui.h4("""Feature Importance
             """),
             ui.p("""Users can view the contribution of each feature they provide. We provided feature importance ranking for three of our algorithms: Naive Bayes, Logistic Regression, and Random Forest. The higher the feature index on the y-axis, the more important the feature is in the course of determining prediction outcome. For kNN and SVM, it is unable to provide the feature importance due to the nature of the algorithm.
             """),
             ui.h3("""PCA Panel
             """),
             ui.p("""On the other panel, we provided a PCA plot for the interpretation of the result as well. Also, by uploading the user's features and choosing the machine learning algorithm, and then click run, the PCA will start running. The user can see if PCA has a similar cluster pattern with the prediction result.
             """)
            ),
        ui.nav(
               'About Us',
                ui.h1("""About Us"""
                ),
                ui.p(
                """We are a group of inspiring bioinformatics scientists from Applied Human Computational Genomics class (Fall 2022) at Georgia Tech. Joseph Luper Tsenum is a second year Bioinformatics PhD Student at Georgia Tech, and he is interested in Artificial intelligence for Multiomics data integration for biomarker discovery & drug discovery/repurposing. Xu Qiu is a second year Bioinformatics MS student at Georgia tech. She previously worked with pan-genome of plants and is fascinated by their diverse structure. Xu is interested in building efficient Bioinformatics pipelines and utilizing various open source tools to solve problems. Zun Wang is a second year Bioinformatics MS student at Georgia Tech, and is interested in Artificial Intelligence for public health data. Right now, she is working on deep learning for MRI image data processing, specifically on Alzheimer's disease."""),
                ui.p("""If you have any questions, please contact us at
                """),
                ui.p("""Engineered Biosystems Building (EBB)
                """),
                ui.p("""Georgia Institute of Technology
                """),
                ui.p("""950 Atlantic Drive Atlanta, GA 30332
                """),
                ui.p("""
                Email Addresses: jtsenum3@gatech.edu, xqiu64@gatech.edu, zwang3311@gatech.edu.""")
                ),
        ui.nav(
                  "References",  
                  ui.h1("References"),
                  ui.p("""1. Zhang, X., Wang, J., Li, J. et al. CRlncRC: a machine learning-based method for cancer-related long noncoding RNA identification using integrated features. BMC Med Genomics 11 (Suppl 6), 120 (2018). https://doi.org/10.1186/s12920-018-0436-9
                  """),
                  ui.p("""2. Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830, 2011.
                  """),  
                  ui.p(
                  """3. Chang W, Cheng J, Allaire J, Sievert C, Schloerke B, Xie Y, Allen J, McPherson J, Dipert A, Borges B (2022). shiny: Web Application Framework for R. R package version 1.7.4, https://shiny.rstudio.com/."""
                    )
                )
    )
app_ui = ui.page_navbar(
    *nav_controls(),
    title="Cancer-Related LncRNA Machine Learning Predictor App (CaRLMaLP R-Shiny App)",
    inverse=True,
    id="navbar_id"
)


def server(input: Inputs, output: Outputs, session: Session):
    @reactive.Effect
    def _():
        print("Current navbar page: ", input.navbar_id())

    @output
    @render.table
    @reactive.event(input.run)
    async def classification_metrics():

        if input.features() is None:
            return "Please upload the feature set (.csv or .tsv files)"
        # if input.pos_target() is None:
        #     return "Please upload the positive target set"
        # if input.neg_target() is None:
        #     return "Please upload the negative target set"

        return run_ML_pipeline('classification_metrics', input)

    @output
    @render.plot
    @reactive.event(input.run)
    async def confusion_matrix():

        if input.features() is None:
            return "Please upload the feature set (.csv or .tsv files)"
        # if input.pos_target() is None:
        #     return "Please upload the positive target set"
        # if input.neg_target() is None:
        #     return "Please upload the negative target set"

        return run_ML_pipeline('confusion_matrix', input)

    @output
    @render.plot
    @reactive.event(input.run)
    async def roc_auc_curve():

        if input.features() is None:
            return "Please upload the feature set (.csv or .tsv files)"
        # if input.pos_target() is None:
        #     return "Please upload the positive target set"
        # if input.neg_target() is None:
        #     return "Please upload the negative target set"

        return run_ML_pipeline('roc_auc_curve', input)

    @output
    @render.plot
    @reactive.event(input.run)
    async def feature_importance():

        if input.features() is None:
            return "Please upload the feature set (.csv or .tsv files)"
        # if input.pos_target() is None:
        #     return "Please upload the positive target set"
        # if input.neg_target() is None:
        #     return "Please upload the negative target set"

        return run_ML_pipeline('feature_importance', input)
    
    #editing##################
    @output
    @render.table
    @reactive.event(input.run)
    async def prediction():

        if input.features() is None:
            return "Please upload the feature set (.csv or .tsv files)"
        # if input.pos_target() is None:
        #     return "Please upload the positive target set"
        # if input.neg_target() is None:
        #     return "Please upload the negative target set"

        return run_ML_pipeline('prediction_result', input)
    ##################

    @output
    @render.plot
    @reactive.event(input.run_pca)
    async def pca():

        if input.features() is None:
            return "Please upload the feature set (.csv or .tsv files)"
        # if input.pos_target() is None:
        #     return "Please upload the positive target set"
        # if input.neg_target() is None:
        #     return "Please upload the negative target set"

        return run_ML_pipeline('pca_plot', input)
    
    



app = App(app_ui, server)
