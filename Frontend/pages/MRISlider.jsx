import React, { useEffect, useState } from "react";

const styles = {
  wrapper: {
    textAlign: "center",
    width: "100%",
    padding: "10px 8px",
    boxSizing: "border-box",
  },
  label: {
    margin: "0 0 8px 0",
    fontSize: "12px",
    fontWeight: 600,
    letterSpacing: "1px",
    color: "#10b981",
    textTransform: "uppercase",
  },
  slider: {
    width: "90%",
    accentColor: "#10b981",
    cursor: "pointer",
    marginBottom: "10px",
  },
  imageContainer: {
    width: "100%",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
  image: {
    maxWidth: "100%",
    maxHeight: "260px",
    objectFit: "contain",
    borderRadius: "6px",
  },
  loading: {
    color: "#7dd3c0",
    fontSize: "13px",
    padding: "40px 0",
  },
  error: {
    color: "#ef4444",
    fontSize: "13px",
    padding: "40px 0",
  },
};

const MRISlider = ({ patientId, plane }) => {
  const [totalSlices, setTotalSlices] = useState(null);
  const [sliceIndex, setSliceIndex] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`http://127.0.0.1:5000/mri-info/${patientId}/${plane}`)
      .then(res => {
        if (!res.ok) {
          throw new Error(`Failed to load ${plane} MRI`);
        }
        return res.json();
      })
      .then(data => {
        setTotalSlices(data.total_slices);
        setSliceIndex(Math.floor(data.total_slices / 2));
      })
      .catch(err => {
        console.error(err);
        setError(err.message);
      });
  }, [patientId, plane]);

  if (error) {
    return <p style={styles.error}>{error}</p>;
  }

  if (totalSlices === null) {
    return <p style={styles.loading}>Loading {plane} MRI...</p>;
  }

  return (
    <div style={styles.wrapper}>
      <h4 style={styles.label}>{plane.toUpperCase()} VIEW</h4>

      <input
        type="range"
        min="0"
        max={totalSlices - 1}
        value={sliceIndex}
        onChange={(e) => setSliceIndex(Number(e.target.value))}
        style={styles.slider}
      />

      <div style={styles.imageContainer}>
        <img
          src={`http://127.0.0.1:5000/mri-slice/${patientId}/${plane}/${sliceIndex}`}
          alt={`${plane} slice`}
          style={styles.image}
          crossOrigin="anonymous"
        />
      </div>
    </div>
  );
};

export default MRISlider;
