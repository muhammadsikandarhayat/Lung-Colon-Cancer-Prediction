import { useState } from "react";
import axios from "axios";
import {
  Upload,
  Brain,
  Activity,
  CheckCircle,
  XCircle,
  AlertCircle,
  AlertTriangle,
} from "lucide-react";
import "./index.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [cancerType, setCancerType] = useState(null);
  const [modelType, setModelType] = useState(null);
  const [currentStep, setCurrentStep] = useState(1); // 1: cancer type, 2: model type, 3: upload
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleCancerTypeSelect = (type) => {
    setCancerType(type);
    setCurrentStep(2);
    setError(null);
  };

  const handleModelTypeSelect = (type) => {
    setModelType(type);
    setCurrentStep(3);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError("Please select an image file first.");
      return;
    }

    if (!cancerType || !modelType) {
      setError("Please complete the setup steps first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(
        `/api/predict/${cancerType}?model_type=${modelType}`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail || "An error occurred during prediction."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setCancerType(null);
    setModelType(null);
    setCurrentStep(1);
    setResult(null);
    setError(null);
  };

  const getResultIcon = () => {
    if (!result) return null;

    if (result.prediction === "Normal") {
      return <CheckCircle size={24} color="#28a745" />;
    } else {
      return <XCircle size={24} color="#dc3545" />;
    }
  };

  const getResultClass = () => {
    if (!result) return "";
    return result.prediction === "Normal" ? "result-normal" : "result-cancer";
  };

  return (
    <div className="App">
      <div className="container">
        <div className="header">
          <h1>
            <Brain
              size={48}
              style={{ marginRight: "16px", verticalAlign: "middle" }}
            />
            Cancer Prediction AI
          </h1>
          <p>
            Upload medical images to get AI-powered predictions for lung and
            colon cancer detection
          </p>
        </div>

        <div className="card">
          {/* Step Indicator */}
          <div className="step-indicator">
            <div className={`step ${currentStep >= 1 ? "active" : ""}`}>
              <span className="step-number">1</span>
              <span className="step-label">Select Cancer Type</span>
            </div>
            <div className={`step ${currentStep >= 2 ? "active" : ""}`}>
              <span className="step-number">2</span>
              <span className="step-label">Choose Model</span>
            </div>
            <div className={`step ${currentStep >= 3 ? "active" : ""}`}>
              <span className="step-number">3</span>
              <span className="step-label">Upload & Predict</span>
            </div>
          </div>

          {/* Step 1: Cancer Type Selection */}
          {currentStep === 1 && (
            <div className="step-content">
              <h3 style={{ marginBottom: "24px", color: "#495057" }}>
                Step 1: Select Cancer Type
              </h3>
              <div className="tabs">
                <button
                  className="tab"
                  onClick={() => handleCancerTypeSelect("lung")}
                >
                  <Activity size={20} />
                  Lung Cancer
                </button>
                <button
                  className="tab"
                  onClick={() => handleCancerTypeSelect("colon")}
                >
                  <Activity size={20} />
                  Colon Cancer
                </button>
              </div>
            </div>
          )}

          {/* Step 2: Model Type Selection */}
          {currentStep === 2 && (
            <div className="step-content">
              <h3 style={{ marginBottom: "24px", color: "#495057" }}>
                Step 2: Choose AI Model
              </h3>
              <p style={{ marginBottom: "16px", color: "#6c757d" }}>
                Selected:{" "}
                <strong>
                  {cancerType.charAt(0).toUpperCase() + cancerType.slice(1)}{" "}
                  Cancer
                </strong>
              </p>
              <div className="tabs">
                <button
                  className="tab"
                  onClick={() => handleModelTypeSelect("cnn")}
                >
                  <Brain size={20} />
                  CNN (Custom)
                </button>
                <button
                  className="tab"
                  onClick={() => handleModelTypeSelect("resnet")}
                >
                  <Brain size={20} />
                  ResNet (Transfer Learning)
                </button>
                <button
                  className="tab"
                  onClick={() => handleModelTypeSelect("efficientnet")}
                >
                  <Brain size={20} />
                  EfficientNet (Transfer Learning)
                </button>
                <button
                  className="tab"
                  onClick={() => handleModelTypeSelect("all")}
                >
                  <Brain size={20} />
                  All Models (Compare)
                </button>
              </div>
              <button
                className="btn btn-secondary"
                style={{ marginTop: "16px" }}
                onClick={() => setCurrentStep(1)}
              >
                ← Back to Cancer Type
              </button>
            </div>
          )}

          {/* Step 3: File Upload and Prediction */}
          {currentStep === 3 && (
            <div className="step-content">
              <h3 style={{ marginBottom: "24px", color: "#495057" }}>
                Step 3: Upload Medical Image
              </h3>
              <div
                style={{
                  marginBottom: "16px",
                  padding: "12px",
                  background: "#f8f9fa",
                  borderRadius: "8px",
                }}
              >
                <p style={{ margin: 0, fontSize: "14px", color: "#495057" }}>
                  <strong>Selected:</strong>{" "}
                  {cancerType.charAt(0).toUpperCase() + cancerType.slice(1)}{" "}
                  Cancer |{" "}
                  {modelType.charAt(0).toUpperCase() + modelType.slice(1)} Model
                </p>
              </div>

              <div className="input-group">
                <label htmlFor="file-input">Select Medical Image</label>
                <input
                  id="file-input"
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="file-input"
                />
                <label htmlFor="file-input" className="file-input-label">
                  <Upload size={20} />
                  {selectedFile
                    ? selectedFile.name
                    : "Choose an image file (JPEG, PNG, etc.)"}
                </label>
              </div>

              {previewUrl && (
                <div className="input-group">
                  <label>Image Preview</label>
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="preview-image"
                  />
                </div>
              )}

              <div style={{ display: "flex", gap: "12px", marginTop: "24px" }}>
                <button
                  className="btn"
                  onClick={handleSubmit}
                  disabled={!selectedFile || isLoading}
                >
                  {isLoading ? (
                    <>
                      <div className="loading"></div>
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Brain size={20} />
                      Predict Cancer
                    </>
                  )}
                </button>

                {selectedFile && (
                  <button className="btn btn-secondary" onClick={resetForm}>
                    Reset
                  </button>
                )}
              </div>

              {error && (
                <div
                  className="result-card"
                  style={{ borderLeftColor: "#dc3545" }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      marginBottom: "12px",
                    }}
                  >
                    <AlertCircle size={20} color="#dc3545" />
                    <strong>Error</strong>
                  </div>
                  <p>{error}</p>
                </div>
              )}

              {error && (
                <div
                  className="result-card"
                  style={{ borderLeftColor: "#dc3545" }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      marginBottom: "12px",
                    }}
                  >
                    <AlertCircle size={20} color="#dc3545" />
                    <strong>Error</strong>
                  </div>
                  <p>{error}</p>
                </div>
              )}

              {result && (
                <div className={`result-card ${getResultClass()}`}>
                  {result.model_type === "all" ? (
                    // Multiple predictions display
                    <div>
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "8px",
                          marginBottom: "16px",
                        }}
                      >
                        <Brain size={24} color="#667eea" />
                        <h3 style={{ margin: 0 }}>
                          Multi-Model Comparison Results
                        </h3>
                      </div>

                      <div style={{ marginBottom: "16px" }}>
                        <p>
                          <strong>Cancer Type:</strong>{" "}
                          {result.cancer_type.charAt(0).toUpperCase() +
                            result.cancer_type.slice(1)}
                        </p>
                        <p>
                          <strong>File:</strong> {result.filename}
                        </p>
                      </div>

                      <div
                        style={{
                          display: "flex",
                          flexDirection: "column",
                          gap: "16px",
                        }}
                      >
                        {result.predictions.map((pred, index) => (
                          <div
                            key={index}
                            style={{
                              border: "1px solid #e9ecef",
                              borderRadius: "8px",
                              padding: "16px",
                              background: "#f8f9fa",
                            }}
                          >
                            {pred.error ? (
                              <div style={{ color: "#dc3545" }}>
                                <strong>
                                  {pred.model_type.toUpperCase()}:
                                </strong>{" "}
                                Error - {pred.error}
                              </div>
                            ) : (
                              <>
                                <div
                                  style={{
                                    display: "flex",
                                    alignItems: "center",
                                    gap: "8px",
                                    marginBottom: "12px",
                                  }}
                                >
                                  {pred.prediction.includes("Cancer") ? (
                                    <AlertTriangle size={20} color="#dc3545" />
                                  ) : (
                                    <CheckCircle size={20} color="#28a745" />
                                  )}
                                  <h4 style={{ margin: 0, fontSize: "16px" }}>
                                    {pred.model_type.toUpperCase()}:{" "}
                                    {pred.prediction}
                                  </h4>
                                </div>

                                <div style={{ marginBottom: "12px" }}>
                                  <p style={{ margin: "4px 0" }}>
                                    <strong>Confidence:</strong>{" "}
                                    {(pred.confidence * 100).toFixed(2)}%
                                  </p>
                                </div>

                                <div>
                                  <p
                                    style={{
                                      margin: "8px 0 4px 0",
                                      fontSize: "14px",
                                    }}
                                  >
                                    <strong>Probabilities:</strong>
                                  </p>
                                  <div style={{ marginBottom: "6px" }}>
                                    <div
                                      style={{
                                        display: "flex",
                                        justifyContent: "space-between",
                                        fontSize: "12px",
                                        marginBottom: "2px",
                                      }}
                                    >
                                      <span>Normal</span>
                                      <span>
                                        {(
                                          pred.probabilities.normal * 100
                                        ).toFixed(1)}
                                        %
                                      </span>
                                    </div>
                                    <div
                                      className="probability-bar"
                                      style={{ height: "6px" }}
                                    >
                                      <div
                                        className="probability-fill probability-normal"
                                        style={{
                                          width: `${
                                            pred.probabilities.normal * 100
                                          }%`,
                                        }}
                                      ></div>
                                    </div>
                                  </div>

                                  <div>
                                    <div
                                      style={{
                                        display: "flex",
                                        justifyContent: "space-between",
                                        fontSize: "12px",
                                        marginBottom: "2px",
                                      }}
                                    >
                                      <span>Cancer</span>
                                      <span>
                                        {(
                                          pred.probabilities.cancer * 100
                                        ).toFixed(1)}
                                        %
                                      </span>
                                    </div>
                                    <div
                                      className="probability-bar"
                                      style={{ height: "6px" }}
                                    >
                                      <div
                                        className="probability-fill probability-cancer"
                                        style={{
                                          width: `${
                                            pred.probabilities.cancer * 100
                                          }%`,
                                        }}
                                      ></div>
                                    </div>
                                  </div>
                                </div>
                              </>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    // Single prediction display
                    <div>
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "8px",
                          marginBottom: "16px",
                        }}
                      >
                        {getResultIcon()}
                        <h3 style={{ margin: 0 }}>
                          Prediction: {result.prediction}
                        </h3>
                      </div>

                      <div style={{ marginBottom: "16px" }}>
                        <p>
                          <strong>Confidence:</strong>{" "}
                          {(result.confidence * 100).toFixed(2)}%
                        </p>
                        <p>
                          <strong>Cancer Type:</strong>{" "}
                          {result.cancer_type.charAt(0).toUpperCase() +
                            result.cancer_type.slice(1)}
                        </p>
                        <p>
                          <strong>Model Type:</strong>{" "}
                          {result.model_type.charAt(0).toUpperCase() +
                            result.model_type.slice(1)}
                        </p>
                        <p>
                          <strong>File:</strong> {result.filename}
                        </p>
                      </div>

                      <div>
                        <p>
                          <strong>Probability Breakdown:</strong>
                        </p>
                        <div style={{ marginBottom: "8px" }}>
                          <div
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              marginBottom: "4px",
                            }}
                          >
                            <span>Normal</span>
                            <span>
                              {(result.probabilities.normal * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="probability-bar">
                            <div
                              className="probability-fill probability-normal"
                              style={{
                                width: `${result.probabilities.normal * 100}%`,
                              }}
                            ></div>
                          </div>
                        </div>

                        <div>
                          <div
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              marginBottom: "4px",
                            }}
                          >
                            <span>Cancer</span>
                            <span>
                              {(result.probabilities.cancer * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="probability-bar">
                            <div
                              className="probability-fill probability-cancer"
                              style={{
                                width: `${result.probabilities.cancer * 100}%`,
                              }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              <button
                className="btn btn-secondary"
                style={{ marginTop: "16px" }}
                onClick={() => setCurrentStep(2)}
              >
                ← Back to Model Selection
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
