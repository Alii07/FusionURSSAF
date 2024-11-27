import streamlit as st
import pandas as pd
import tempfile
import os
from Bases import AnomalyDetection
from Taux import Taux


versement_mobilite = { '87005' : 1.80 }
models_info = {
    '6000': {
        'type' : 'joblib',
        'model': './Modèles/Taux/6000.pkl',
        'numeric_cols': ['Rub 6000',  '6000Taux'],
        'categorical_cols': ['Frontalier'],
        'target_col': 'anomalie_frontalier'
    },
    '6002': {
        'type' : 'joblib',
        'model': './Modèles/Taux/6002.pkl',
        'numeric_cols': ['Rub 6002',  '6002Taux'],
        'categorical_cols': ['Region'],
        'target_col': 'anomalie_alsace_moselle'
    },
    '6082': {
        'type': 'joblib',
        'model': './Modèles/Taux/6082.pkl',
        'numeric_cols': ['Rub 6082', '6082Taux'],
        'categorical_cols': ['Statut de salariés', 'Frontalier'],
        'target_col': 'anomalie_csg'
    },
    '6084': {
        'type': 'joblib',
        'model': './Modèles/Taux/6084.pkl',
        'numeric_cols': ['Rub 6084', '6084Taux'],
        'categorical_cols': ['Statut de salariés', 'Frontalier'],
        'target_col': 'anomalie_crds'
    },
    '7001': {
            'type' : 'joblib',
            'model': './Modèles/Taux/7001.pkl',
            'numeric_cols': ['Matricule', 'Absences par Jour', 'Absences par Heure', 'PLAFOND CUM', 'ASSIETTE CUM', 'MALADIE CUM', '7001Base', '7001Taux 2', '7001Montant Pat.'],
            'categorical_cols': ['Statut de salariés'],
            'target_col': 'anomalie_maladie_reduite'
        },
    '7002': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7002_cases.pkl',  # Utilisez le chemin vers votre modèle
        'numeric_cols': ['SMIC M CUM', '7002Taux 2', 'ASSIETTE CUM'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': 'anomalie_maladie_diff'
    },

    '7010': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7010.pkl',
        'numeric_cols': ['Rub 7010',  '7010Taux 2','7010Taux' ,'Effectif'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': 'Anomalie_7010'
    },

    '7015': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7015.pkl',
        'numeric_cols': ['Rub 7015','7015Taux', '7015Taux 2' ,'Effectif'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': 'Anomalie_7015'
    },

    '7020': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7020.pkl',
        'numeric_cols': ['Rub 7020',  '7020Taux 2' ,'Effectif'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': 'anomalie_fnal'
    },

    '7025': {
            'type': 'joblib',
            'model': './Modèles/Taux/7025.pkl',
            'numeric_cols': ['7025Taux 2', 'ASSIETTE CUM', 'PLAFOND CUM'],
            'categorical_cols': ['Statut de salariés'],
            'target_col': '7025Taux 2'
        },
    '7030': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7030.pkl',
        'numeric_cols': ['PLAFOND CUM', 'ASSIETTE CUM','7030Taux 2', 'Rub 7030'],
        'categorical_cols': [],
        'target_col': 'anomalie_allocation_reduite'
    },
    '7035': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7035.pkl',
        'numeric_cols': ['Rub 7035','7035Taux 2'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': '7035 Fraud'
    },
    '7040': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7040.pkl',
        'numeric_cols': ['Effectif', '7040Taux 2' ,'Rub 7040'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': 'anomalie_7040taux 2'
    },
    '7045': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7045.pkl',
        'numeric_cols': ['Effectif', '7045Taux 2'],
        'categorical_cols': ['Etablissement'],
        'target_col': 'anomalie_transport'
    },
    '7050': {
        'type' : 'joblib',
        'model': './Modèles/Taux/7050.pkl',
        'numeric_cols': ['Effectif', '7050Taux 2'],
        'categorical_cols': ['Etablissement'],
        'target_col': 'anomalie_cotisation_accident'
    }
}

model_configs = {
    "6081": {
        "classification_model": "Modèles/Bases/classification_model_6081_new.pkl",
        "regression_models": {
            (0, 0): "Modèles/Bases/regression_model_6081_(0, 0).pkl",
            (0, 1): "Modèles/Bases/regression_model_6081_(0, 1).pkl",
            (1, 1): "Modèles/Bases/regression_model_6081_(1, 1).pkl",
        },
        "numeric_cols": {
            (0, 0): ["4*PLAFOND CUM", "Cumul d'assiette ( Mois courant inclus) (/102)"],
            (0, 1): ["4*PLAFOND CUM", "Cumul d'assiette ( Mois courant inclus) (/102)"],
            (1, 1): ["Tranche C pre"],
        },
        "target_col": "6081Base",
        "apprenti_status": ["Apprenti (B.C)", "Apprenti (W.C)"],
        "threshold_margin": 0.01,
    },
    "6085": {
        "classification_model": "Modèles/Bases/classification_model_6085_new.pkl",
        "regression_models": {
            (0, 0): "Modèles/Bases/regression_model_6085_(0, 0).pkl",
            (0, 1): "Modèles/Bases/regression_model_6085_(0, 1).pkl",
            (1, 1): "Modèles/Bases/regression_model_6085_(1, 1).pkl",
        },
        "numeric_cols": {
            (0, 0): ["4*PLAFOND CUM", "Cumul d'assiette ( Mois courant inclus) (/102)"],
            (0, 1): ["4*PLAFOND CUM", "Cumul d'assiette ( Mois courant inclus) (/102)"],
            (1, 1): ["Tranche C pre"],
        },
        "target_col": "6085Base",
        "apprenti_status": ["Apprenti (B.C)", "Apprenti (W.C)"],
        "threshold_margin": 0.01,
    },
    "6082": {
        "path": "Modèles/Bases/6082.pkl",
        "cases": {
            "Cas 1": {"feature_col": "Plafond CUM", "target_col": "6082Base"},
            "Cas 2": {"feature_col": "1001Montant Sal.", "target_col": "6082Base"},
        },
        "threshold_margin": 0.01,
    },
    "6084": {
        "path": "Modèles/Bases/6084.pkl",
        "cases": {
            "Cas 1": {"feature_col": "Plafond CUM", "target_col": "6084Base"},
            "Cas 2": {"feature_col": "1001Montant Sal.", "target_col": "6084Base"},
        },
        "threshold_margin": 0.01,
    },
    
    "7002": {
        "path": "Modèles/Bases/7002.pkl",
        "target_col": "7002Base",
        "threshold_margin": 0.01,
    },
    
    "7015": {
        "classification_model": "Modèles/Bases/classification_assiette_plafond.joblib",
        "regression_models": {
            0: "Modèles/Bases/model_label_0.joblib",
            1: "Modèles/Bases/model_label_1.joblib",
        },
        "classification_features": ["Assiette cum", "PLAFOND CUM"],
        "regression_features": {
            0: ["PLAFOND CUM", "CUM Base 7015 precedente"],
            1: ["Assiette cum", "CUM Base 7015 precedente"],
        },
        "required_columns": ["Assiette cum", "PLAFOND CUM", "CUM Base precedente", "7015Base"],
        "target_col": "7015Base",
        "threshold_margin": 0.01,
    },
    "7025": {
        "path": "Modèles/Bases/7025.pkl",
        "numeric_cols": ["7025Base", "Base CUM M-1", "Brut CUM", "Plafond CUM", "Total Brut"],
        "categorical_cols": ["Cluster"],
        "threshold_margin": 0.01,
    },
}



simple_models = ["7001", "7020", "7030", "7035", "7040", "7045", "7050"]

def generate_base_anomalies_report_streamlit(df, simple_models, model_configs):
    """
    Génère un rapport des anomalies détectées pour les bases, adapté à Streamlit.
    """
    base_anomaly_detector = AnomalyDetection(model_configs)
    base_anomaly_detector.load_models()

    # Conteneur pour les rapports d'anomalies
    base_reports = {}
    error_log = []  # Stockage des erreurs pour un éventuel débogage

    # Détection des anomalies simples
    for model_name in simple_models:
        try:
            report = base_anomaly_detector.detect_anomalies_simple_comparison(df.copy(), model_name)
            if report is not None:
                base_reports[model_name] = report
        except Exception as e:
            error_log.append(f"Erreur pour le modèle simple {model_name}: {e}")

    # Détection des anomalies avancées
    for model_name in model_configs.keys():
        try:
            df_preprocessed = base_anomaly_detector.preprocess_data(df.copy(), model_name)
            
            if isinstance(model_configs[model_name].get('numeric_cols'), dict):
                # Gestion spécifique pour les modèles avancés (ex : 6081, 6085)
                numeric_cols = []
                for subset_cols in model_configs[model_name]['numeric_cols'].values():
                    numeric_cols.extend(subset_cols)  # Ajouter toutes les colonnes pour chaque sous-ensemble
                required_columns = list(set(numeric_cols)) + model_configs[model_name].get('categorical_cols', [])
            else:
                # Gestion standard (ex : modèles simples)
                required_columns = model_configs[model_name].get('numeric_cols', []) + model_configs[model_name].get('categorical_cols', [])

            # Vérifier les colonnes manquantes
            missing_columns = [col for col in required_columns if col not in df_preprocessed.columns]
            if missing_columns:
                error_log.append(f"Le modèle {model_name} manque des colonnes : {', '.join(missing_columns)}")
                continue

            # Détection des anomalies
            report = base_anomaly_detector.detect_anomalies(df_preprocessed, model_name)
            if report is not None:
                base_reports[model_name] = report
        except Exception as e:
            error_log.append(f"Erreur pour le modèle avancé {model_name}: {e}")

    # Combiner les rapports
    try:
        base_combined_report = base_anomaly_detector.combine_reports(base_reports)
    except Exception as e:
        error_log.append(f"Erreur lors de la combinaison des rapports : {e}")
        return None, error_log

    # Générer les lignes du rapport des bases
    report_lines = []
    for index, row in base_combined_report.iterrows():
        matricule = row['Matricule'] if 'Matricule' in row else f"Ligne {index}"
        model_name = row['nom_du_modèle'] if 'nom_du_modèle' in row else "Modèle inconnu"
        report_lines.append(f"Nous avons détecté pour le Matricule {matricule} une anomalie dans la cotisation : {model_name}\n")

    return report_lines, error_log



def detect_taux_anomalies_streamlit(df):
    """
    Génère un rapport des anomalies détectées pour les taux, adapté à Streamlit.
    """
    taux_anomaly_detector = Taux(models_info, versement_mobilite)
    error_log = []  # Stockage des erreurs pour un éventuel débogage

    try:
        anomalies, _ = taux_anomaly_detector.detect_anomalies(df)
    except Exception as e:
        error_log.append(f"Erreur lors de la détection des anomalies des taux: {e}")
        return None, error_log

    # Générer les lignes du rapport des taux
    report_lines = []
    for index, details in anomalies.items():
        matricule = df.loc[index, 'Matricule'] if 'Matricule' in df.columns else f"Ligne {index}"
        filtered_models = [model for model in details if model in models_info.keys()]
        if filtered_models:
            report_lines.append(f"Nous avons détecté pour le Matricule {matricule} une anomalie dans la cotisation : {', '.join(filtered_models)}\n")

    return report_lines, error_log


def merge_anomalies_reports_streamlit(base_lines, taux_lines):
    """
    Combine les rapports des anomalies des bases et des taux en triant les matricules par ordre croissant.
    """
    anomalies_by_matricule = {}

    def extract_anomalies(lines):
        for line in lines:
            if line.strip().startswith("Nous avons détecté pour le Matricule"):
                # Extraire le matricule et les modèles
                parts = line.split("une anomalie dans la cotisation :")
                matricule = parts[0].split("Matricule")[-1].strip()
                models = [model.strip() for model in parts[1].split(",")]
                if matricule not in anomalies_by_matricule:
                    anomalies_by_matricule[matricule] = set()
                anomalies_by_matricule[matricule].update(models)

    # Extraire les anomalies des deux rapports
    extract_anomalies(base_lines)
    extract_anomalies(taux_lines)

    # Trier les anomalies par matricule
    sorted_anomalies = dict(sorted(anomalies_by_matricule.items(), key=lambda x: x[0]))

    # Générer le rapport fusionné
    combined_lines = ["=== Rapport combiné des anomalies ===\n\n"]
    for matricule, models in sorted_anomalies.items():
        models_list = ", ".join(sorted(models))  # Trier les modèles pour un format cohérent
        combined_lines.append(f"Nous avons détecté pour le Matricule {matricule} une anomalie dans la cotisation : {models_list}\n")

    return "".join(combined_lines)



def main():
    st.title("Détection des Anomalies dans les Bases et les Taux")
    st.write("Chargez votre fichier CSV pour détecter les anomalies et générer un rapport combiné.")

    # Upload du fichier CSV
    uploaded_file = st.file_uploader("Chargez votre fichier CSV", type=["csv"])
    if uploaded_file is not None:
        # Charger le fichier CSV
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            st.success(f"Fichier chargé avec succès ! Dimensions : {df.shape}")
            st.write("Aperçu des données :")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            return

        # Bouton pour lancer l'analyse
        if st.button("Lancer la détection des anomalies"):
            with st.spinner("Analyse en cours..."):
                try:
                    # Détecter les anomalies des bases
                    base_lines, base_errors = generate_base_anomalies_report_streamlit(df, simple_models, model_configs)

                    # Détecter les anomalies des taux
                    taux_lines, taux_errors = detect_taux_anomalies_streamlit(df)

                    # Fusionner les rapports
                    if base_lines is not None and taux_lines is not None:
                        combined_report_content = merge_anomalies_reports_streamlit(base_lines, taux_lines)

                        # Télécharger le rapport combiné
                        st.success("Rapport combiné généré avec succès.")
                        st.download_button(
                            label="Télécharger le rapport combiné",
                            data=combined_report_content,
                            file_name="rapport_anomalies_combiné.txt",
                            mime="text/plain"
                        )

                    # Optionnel : Afficher les erreurs
                    if base_errors or taux_errors:
                        for error in base_errors + taux_errors:
                            st.text(error)

                except Exception as e:
                    st.error(f"Erreur lors de l'analyse : {e}")


if __name__ == "__main__":
    main()
