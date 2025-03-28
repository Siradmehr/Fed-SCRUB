def compute_mia_score(self, model):
    X_f, y_f = self.get_loss_dataset(model, self.forgetloader, label=1)
    X_t, y_t = self.get_loss_dataset(model, self.valloader, label=0)

    X = np.vstack([X_f, X_t])
    y = np.concatenate([y_f, y_t])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []

    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression()
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], preds)
        accs.append(acc)

    return np.mean(accs)