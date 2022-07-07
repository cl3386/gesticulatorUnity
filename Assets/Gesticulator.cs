using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Python.Runtime;
// using System.Collections;
// using System.Collections.Generic;
using UnityEngine;
using System.IO;


public class Gesticulator : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        //gesticulator inference
        Main();

        //copy bvh file(generated at Project Folder) into Assets folder
        // copyBvh();
    }

    // Update is called once per frame
    void Update()
    {

    }

    static void copyBvh()
    {

        File.Copy(@"C:/Users/CY/Desktop/desktop/pythonRuntime/temp.bvh", @"C:/Users/CY/Desktop/desktop/pythonRuntime/Assets/temp.bvh");

    }
    static void Main()


    {
        Runtime.PythonDLL = @"C:\Users\CY\AppData\Local\Programs\Python\Python38\python38.dll";

        var PYTHON_HOME = Environment.ExpandEnvironmentVariables(@"C:\Users\CY\AppData\Local\Programs\Python\Python38");

        // Python 홈 설정.
        PythonEngine.PythonHome = PYTHON_HOME;
        // 모듈 패키지 패스 설정.
        PythonEngine.PythonPath = string.Join(

            Path.PathSeparator.ToString(),
            new string[] {
                  PythonEngine.PythonPath,
                    // pip하면 설치되는 패키지 폴더.
                     Path.Combine(PYTHON_HOME, @"Lib\site-packages"),  
                    // 개인 패키지 폴더
                    @"C:\Users\CY\Desktop\desktop\pythonRuntime\Assets\pythonCode",
                    // @"C:\Users\CY\Desktop\desktop\pythonRuntime\Assets\pythonCode\demo",
                    @"C:\Users\CY\Desktop\desktop\pythonRuntime\Assets\pythonCode\gesticulator",
                    @"C:\Users\CY\Desktop\desktop\pythonRuntime\Assets\pythonCode\gesticulator\visualization"
            }
        );
        // Python 엔진 초기화
        PythonEngine.Initialize();

        // Global Interpreter Lock을 취득
        using (Py.GIL())
        {

            // dynamic demo = Py.Import("demo.democs");   // It uses  PythonEngine.PythonPath    
            // dynamic motionPythonArray = demo.main();


            //  dynamic pysys = Py.Import("sys");   // import sys module from  PythonEngine.PythonPath 
            //     dynamic pySysPath = pysys.path;
            //     string[] sysPathArray = (string[])pySysPath;    // About conversion: https://csharp.hotexamples.com/site/file?hash=0x7a3b7b993fab126a5a205be68df1c82bd87e4de081aa0f5ad36909b54f95e3d7&fullName=&project=pythonnet/pythonnet

            //     List<string> sysPath = ((string[])pySysPath).ToList<string>();
            //     Debug.Log(pysys.path);
            //     Debug.Log(pySysPath);
            //     Debug.Log(sysPath);

            // // 개인 패키지 폴더의 Demo/demo.py 읽기.   
            // dynamic demo = Py.Import("demo.demo");   // It uses  PythonEngine.PythonPath    
            // dynamic motionPythonArray = demo.main();

            // PyList motionPyList = PyList.AsList(motionPythonArray);
            // //  PyList motionPyList = PyList.AsList
            // Debug.Log("\n\n Print Python List  in Console#\n");

            // for (int i = 0; i < 520; i++)
            // {
            //     Debug.LogFormat("{0}: \n", i);
            //     for (int j = 0; j < 45; j++)
            //     {
            //         //motionArray[i,j] = (float)motionPythonList[i][j];
            //         Debug.Log($"{motionPyList[i][j]} \t");



            //     }

            //     Debug.Log("\n");

            // }


            string text = "Deep learning is an algorithm inspired by how the human brain works, and as a result it's an algorithm which has no theoretical limitations on what it can do. The more data you give it and the more computation time you give it, the better it gets. The New York Times also showed in this article another extraordinary result of deep learning which I'm going to show you now. It shows that computers can listen and understand.";
            dynamic librosa = Py.Import("librosa");  // import a package
            dynamic audio_sample_rate = librosa.load(@"C:\Users\CY\Desktop\desktop\pythonRuntime\Assets\pythonCode\demo\input\jeremy_howard.wav");

            var audio = audio_sample_rate[0];
            // np.array audio =  audio_sample_rate[0];  or dynamic audio = audio_sample_rate[0]
            int sample_rate = audio_sample_rate[1];

            dynamic democs = Py.Import("demo.democs");

            dynamic motionPythonArray = democs.main(audio, text, sample_rate);
            PyList motionPyList = PyList.AsList(motionPythonArray);


            // motionPyList = PyList.AsList(motionPythonArray);

            Debug.Log("\n\n Print Python List  in Console:  Passing input to gesticulator from csharp\n");


            for (int i = 0; i < 520; i++)
            {
                Console.WriteLine("{0}: \n", i);
                for (int j = 0; j < 45; j++)
                {
                    //motionArray[i,j] = (float)motionPythonList[i][j];
                    Debug.Log($"{motionPyList[i][j]} \t");



                }

                Debug.Log("\n");

            }
            // using GIL( Py.GIL() )

            // python 환경을 종료한다.
            PythonEngine.Shutdown();

        }
    }
    }


