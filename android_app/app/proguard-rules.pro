# Keep TensorFlow Lite classes
-keep class org.tensorflow.** { *; }
-keepclassmembers class * {
    @org.tensorflow.lite.schema.Required *;
}
