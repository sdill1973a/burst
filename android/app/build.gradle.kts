plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.jetbrains.kotlin.android)
}

android {
    androidResources { noCompress += listOf("gguf", "bin") }
    namespace = "com.gtc.burst.app"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.gtc.burst"

        minSdk = 33
        targetSdk = 36

        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
        // Test hooks (intent extras that drive the radio, the admin broadcast): off unless -PburstTestHooks=true.
        buildConfigField("boolean", "TEST_HOOKS",
            ((project.findProperty("burstTestHooks") as String?) == "true").toString())
    }
    buildFeatures { buildConfig = true }

    buildTypes {
        debug {
            if (project.hasProperty("burstFresh")) applicationIdSuffix = ".fresh"   // a clean first-run install beside the real one
            isMinifyEnabled = false
            isShrinkResources = false
            proguardFiles(
                getDefaultProguardFile("proguard-android.txt"),
                "proguard-rules.pro"
            )
        }
        release {
            if (project.hasProperty("burstFresh")) applicationIdSuffix = ".fresh"   // test the shipped (minified) build beside the real one
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(libs.bundles.androidx)
    implementation(libs.material)

    implementation(project(":lib"))

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
