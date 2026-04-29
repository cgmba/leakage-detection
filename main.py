import shap
# explain the model's predictions using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
# Visualize the first prediction's explanation
shap.summary_plot(shap_values, X_test)
