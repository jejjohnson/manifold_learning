from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Naive Random forest classification scheme
def rf_naive(X, y, img, train_prct=0.10):
    """Implements a Naive RF Classifier on a Hyperspectral Image
    Parameters:
    -----------
    X data set: (NxD) where N are the features
    and D is the dimension
    
    img: an image that is (MxNxD)
    img_dimred: an image that is (MxNxd)
    """
    #--------------------------------
    # Training and Testing Data Split
    #--------------------------------
    train_sample_size = train_prct

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_sample_size,
        random_state=1234)
    
    #--------------------------------------------------
    # Implement the Random Forest Classification Method
    #--------------------------------------------------
    # Initialize our model with 500 trees
    rf_param = RandomForestClassifier(
                n_estimators=500, oob_score=True)

    print('Using Random Forest Parameters to train the model...')
    t0 = time()
    # Fit our model to training Data
    rf_model = rf_param.fit(X_train, y_train)
    print('Done in {s:0.3f}s.'.format(
        s=time()-t0))

    print ('Our OOB Prediction accuracy is: {oob}'.format(
            oob=rf_model.oob_score_*100))

    print('Using Random Forest model on the testing data...')
    t0 = time()
    # Test the model on the testing data
    y_pred = rf_model.predict(X_test)

    print('Done in {s:0.3f}s.'.format(
        s=time()-t0))
    
    #--------------------
    # Accuracy Statistics
    #--------------------
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    #--------------------------------
    # Classification for entire image
    #--------------------------------
    

    print('Using the RF model to predict the rest of the image classes...')
    t0 = time()
    rf_class_img = rf_model.predict(img)
    print('Done in {s:0.3f}s.'.format(
        s=time()-t0))
    
    return rf_class_img, cm, cr