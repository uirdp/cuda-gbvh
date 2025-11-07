// ファイル入出力のためのライブラリ
#pragma once

#include "scene.h"
#include "paramerters.h"
#include <string>

/*
 * パラメタファイルを読み、その情報を param に格納する 
 *   - .obj ファイルの読み込みはこの関数からは行わない。
 *   - 戻り値は正常終了なら 0、エラーなら RT_ERROR
 */
int RT_ReadParamFile(const std::string& filename, InputParameter& param, Scene &scene);

/* 
 * .objファイルを読み、その情報を scene に格納する
 *   - 戻り値は正常終了なら 0、エラーなら RT_ERROR
 */
int RT_ReadObjFile(const InputParameter& param, Scene &scene);

/* 
 * 材質ファイルを読み、その情報を scene に格納する
 *   - RT_ReadObjFile から呼ばれる。
 *   - 戻り値は正常終了なら 0、エラーなら RT_ERROR
 */
int RT_ReadMaterialFile(const std::string& filename,
			const std::string& dirname, const std::string& dirname_alt,
			Scene &scene);
