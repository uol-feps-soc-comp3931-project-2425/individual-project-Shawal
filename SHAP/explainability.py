@ -20,6 +20,7 @@ pretty_names = {
    "issue_responded_hour": "Response Sent Hour",
    "issue_responded_dayofweek": "Response Sent Day",
    "Issue_reported_at_hour": "Issue Reported Hour",
    "issue_responded_hour_missing": "Response Sent Hour (Missing)",

    "Agent_Shift_Morning": "Agent: Morning Shift",
    "Agent_Shift_Evening": "Agent: Evening Shift",
@ -88,6 +89,22 @@ def plot_shap_group(features, shap_values, X, title, model_name):
    plt.savefig(f"SHAP_{model_name.replace(' ', '_')}_{title.replace(' ', '_')}.png", dpi=300)
    plt.close()

def plot_shap_bar(shap_values, X, model_name):
    X_display = X.rename(columns=pretty_names)
    bubblegum = "#f47ac1"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_display, plot_type="bar", show=False, max_display=10)
    ax = plt.gca()
    for patch in ax.patches:
        patch.set_facecolor(bubblegum)
    plt.title(f"Top 10 SHAP Features – {model_name}", fontsize=14)
    plt.xlabel("Average SHAP Value (Impact on Prediction)", fontsize=12, labelpad=8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"SHAP_BAR_{model_name.replace(' ', '_')}.png", dpi=300)
    plt.close()

#run SHAP on any model
def explain_model(model_name):
    print(f"Explaining {model_name}...")
