#include "includes/fileio.h"
#include "includes/statistics.h"
#include "includes/spline.h"
#include "includes/external/glm/vec3.hpp"
#include <ctype.h>
#include <opencv2/opencv.hpp>

#define MAX_ERRS  10

using std::string;
using glm::vec3;
using glm::ivec3;
/* 下の１行は、本来は Matrial.cpp に入るべき */
Material Material::default_material("default_material");

/* 下の scan_XXX 関数群は、読み取りに成功した場合に true を返す */
static bool scan_string(const char * &p, string &s) {
    while( *p == ' ' || *p == '\t' ) p++;
    if( *p == 0 || *p == '#' || *p == '\r' || *p == '\n' ) {
	/* end of line */
	return false;
    }
    while( *p != 0 && *p != ' ' && *p != '\t' && 
	   *p != '\r' && *p != '\n' && *p != '#' ) s += *p++;
    return true;
}

static bool scan_integer(const char * &p, int &val) {
    while( *p == ' ' || *p == '\t' ) p++;
    if( sscanf(p, "%d", &val) != 1 )  return false;
    while( isdigit(*p) || *p == '+' || *p == '-' ) p++;
    return true;
}

static bool scan_float(const char * &p, float &val) {
    while( *p == ' ' || *p == '\t' ) p++;
    if( sscanf(p, "%f", &val) != 1 )  return false;
    while( isdigit(*p) || *p == '.' || *p == 'e' || *p == 'E' || 
	   *p == '+' || *p == '-' ) p++;
    return true;
}

static bool scan_slashed_three_integers(const char * &p, glm::ivec3 &val) {
    val = glm::ivec3(0,0,0);
    if( !scan_integer(p, val.x) ) return false;
    while( *p == ' ' || *p == '\t' ) p++;
    if( *p == '/' ) {
	p++;
	scan_integer(p, val.y);
    }
    while( *p == ' ' || *p == '\t' ) p++;
    if( *p == '/' ) {
	p++;
	scan_integer(p, val.z);
    }
    return true;
}

/* ファイルパス名からディレクトリ名を取り出す */
static string getDirName(const string& filename) {
    int dpos = filename.find_last_of("/\\");
    return dpos >= 0 ? filename.substr(0, dpos+1) : "";
};

int 
RT_ReadParamFile(const string& _filename, InputParameter& param, Scene& scene)
{
    const char *filename = _filename.c_str();
    int  err = 0;
    int  line = 0;
    char buf[RT_BUFLEN];

    FILE  *fp = fopen(filename, "r");;
    if( !fp ) {
	fprintf(stderr, "ERROR: ");
	perror(filename);
	return RT_ERROR;
    }
    param.file_name = filename;

    scene.view_scenario.resize(1);

    while( fgets(buf, RT_BUFLEN, fp) ) {
	line++;
	if( err >= MAX_ERRS )  break;
	if( strlen(buf) == RT_BUFLEN - 1 ) {
	    fprintf(stderr, "ERROR: %s: Line %d: Too long input line\n", 
		    filename, line);
	    ++err;
	}
	
	const char *p = buf;
	string header;
	bool argErr = false;
	if( !scan_string(p, header) ) continue;

	if( header == "objfile" ) {
	    argErr = !scan_string(p, param.obj_file);
	} else if( header == "imgsize" ) {
	    argErr = (!scan_integer(p, param.image_size.x) ||
		      !scan_integer(p, param.image_size.y));
	} else if( header == "frame" ) {
	    argErr = (!scan_integer(p, param.start_frame) ||
		      !scan_integer(p, param.end_frame));
	    if( param.end_frame - param.start_frame > 0 ) {
		// 動画の場合は、繰り返し回数を１にする
		param.render_repeat = 1;
	    }
	} else if( header == "viewpoint" ) {
	    argErr = (!scan_float(p, param.view_point.x) ||
		      !scan_float(p, param.view_point.y) ||
		      !scan_float(p, param.view_point.z));
	} else if( header == "viewpointa" ) {
	    int fr;
	    vec3 viewpoint;
	    argErr = (!scan_integer(p, fr) ||
		      !scan_float(p, viewpoint.x) ||
		      !scan_float(p, viewpoint.y) ||
		      !scan_float(p, viewpoint.z));
	    if( scene.view_scenario.size() < fr + 1 )
		scene.view_scenario.resize(fr + 1);
	    scene.view_scenario[fr].push_back(ViewAction(ViewAction::VIEWPOINT,
							 viewpoint));
	} else if( header == "lookat" ) {
	    argErr = (!scan_float(p, param.look_at.x) ||
		      !scan_float(p, param.look_at.y) ||
		      !scan_float(p, param.look_at.z));
	} else if( header == "lookata" ) {
	    int fr;
	    vec3 lookat;
	    argErr = (!scan_integer(p, fr) ||
		      !scan_float(p, lookat.x) ||
		      !scan_float(p, lookat.y) ||
		      !scan_float(p, lookat.z));
	    if( scene.view_scenario.size() < fr + 1 )
		scene.view_scenario.resize(fr + 1);
	    scene.view_scenario[fr].push_back(ViewAction(ViewAction::LOOKAT,
							 lookat));
	} else if( header == "viewup" ) {
	    argErr = (!scan_float(p, param.view_up.x) ||
		      !scan_float(p, param.view_up.y) ||
		      !scan_float(p, param.view_up.z));
	    param.view_up = glm::normalize(param.view_up);
	} else if( header == "viewupa" ) {
	    int fr;
	    vec3 viewup;
	    argErr = (!scan_integer(p, fr) ||
		      !scan_float(p, viewup.x) ||
		      !scan_float(p, viewup.y) ||
		      !scan_float(p, viewup.z));
	    viewup = normalize(viewup);
	    if( scene.view_scenario.size() < fr + 1 )
		scene.view_scenario.resize(fr + 1);
	    scene.view_scenario[fr].push_back(ViewAction(ViewAction::VIEWUP,
							 viewup));
	} else if( header == "fovy" ) {
	    argErr = !scan_float(p, param.camera_fov);
	} else if( header == "shadowing" ) {
	    argErr = !scan_float(p, param.shadowIntensity);
	} else if( header == "background" ) {
	    argErr = (!scan_float(p, param.background_color.r) ||
		      !scan_float(p, param.background_color.g) ||
		      !scan_float(p, param.background_color.b));
	} else if( header == "ambient" ) {
	    argErr = !scan_float(p, param.ambient_intensity);
	} else if( header == "dirlight" ) {
	    vec3 color, dir;
	    argErr = (!scan_float(p, dir.x) ||
		      !scan_float(p, dir.y) ||
		      !scan_float(p, dir.z) ||
		      !scan_float(p, color.r) ||
		      !scan_float(p, color.g) ||
		      !scan_float(p, color.b));
	    param.lights.push_back(Light(normalize(dir), color));
	} else if( header == "pointlight" ) {
	    vec3 color, pos, att;
	    argErr = (!scan_float(p, pos.x) ||
		      !scan_float(p, pos.y) ||
		      !scan_float(p, pos.z) ||
		      !scan_float(p, color.r) ||
		      !scan_float(p, color.g) ||
		      !scan_float(p, color.b) ||
		      !scan_float(p, att[0]) ||
		      !scan_float(p, att[1]) ||
		      !scan_float(p, att[2]));
	    param.lights.push_back(Light(pos, color, att));
	} else if( header == "bump" ) {
	    argErr = !scan_float(p, param.bump_strength);
	} else {
	    fprintf(stderr, "ERROR: %s: Line %d: Unknown line header `%s'\n", 
		    filename, line, header.c_str());
	    ++err;
	    break;
	}

	if( argErr ) {
	    fprintf(stderr, "ERROR: %s: Line %d: %s: Too few arguments\n", 
		    filename, line, header.c_str());
	    ++err;
	}

	string excess;
	if( scan_string(p, excess) ) {
	    fprintf(stderr, "ERROR: %s: Line %d: %s: Excess argument `%s'\n", 
		    filename, line, header.c_str(), excess.c_str());
	    ++err;
	}
    }

    if( param.obj_file.empty() ) {
	fprintf(stderr, "ERROR: No `objfile' record - It seems that \"%s\" is not a parameter file.\n",
		filename);
	++err;
    }

    fclose(fp);
    return err > 0 ? RT_ERROR : 0;
}

int 
RT_ReadMaterialFile(const string& _filename,
		    const string& _dirname, const string& _dirname_alt,
		    Scene& scene)
{
    const char *filename = _filename.c_str();
    string dirname = _dirname;
    int  err = 0;
    int  line = 0;
    char buf[RT_BUFLEN];
    Material *current = &Material::default_material;

    FILE  *fp = fopen((dirname + filename).c_str(), "r");
    if( !fp ) {
	dirname = _dirname_alt;
	fp = fopen((dirname + filename).c_str(), "r");
	if( !fp ) {
	    fprintf(stderr, "ERROR: ");
	    perror(filename);
	    return RT_ERROR;
	}
    }

    while( fgets(buf, RT_BUFLEN, fp) ) {
	line++;
	if( err >= MAX_ERRS )  break;
	if( strlen(buf) == RT_BUFLEN - 1 ) {
	    fprintf(stderr, "ERROR: %s: Line %d: Too long input line\n", 
		    filename, line);
	    ++err;
	}
	
	const char *p = buf;
	string header;
	bool argErr = false;
	if( !scan_string(p, header) ) continue;

	if( header == "newmtl" ) {
	    string name;
	    if( !(argErr = !scan_string(p, name)) ) {
		current = new Material(name);
		scene.materials[name] = current;
	    }
	} else if( header == "Ka" ) {
	    argErr = (!scan_float(p, current->ambient.r) ||
		      !scan_float(p, current->ambient.g) ||
		      !scan_float(p, current->ambient.b));
	} else if( header == "Kd" ) {
	    argErr = (!scan_float(p, current->diffuse.r) ||
		      !scan_float(p, current->diffuse.g) ||
		      !scan_float(p, current->diffuse.b));
	} else if( header == "Ks" ) {
	    argErr = (!scan_float(p, current->specular.r) ||
		      !scan_float(p, current->specular.g) ||
		      !scan_float(p, current->specular.b));
	} else if( header == "Ns" ) {
	    argErr = !scan_float(p, current->exponent);
	} else if( header == "reflectance" ) {
	    argErr = !scan_float(p, current->reflectance);
	} else if( header == "transparency" || header == "transmission" ) {
	    argErr = !scan_float(p, current->transparency);
	} else if( header == "Ni" ) {
	    argErr = !scan_float(p, current->refraction_index);
	} else if( header == "map_Ka" || 
		   header == "map_Kd" ||
		   header == "map_Ks" ||
		   header == "map_bump" ||
		   header == "map_Bump" ) {
	    string name;
	    if( !(argErr = !scan_string(p, name)) ) {
		std::map<string,Texture*>::iterator m = scene.textures.find(name);
		Texture *texture = NULL;
		if( m != scene.textures.end() ) texture = m->second;
		else {
		    int fmt = (header == "map_bump" || 
			       header == "map_Bump") ? 0 : 1;
		    cv::Mat img = cv::imread(name, fmt);
		    if( !img.data &&
			!(img = cv::imread(dirname + name, fmt)).data ) {
			fprintf(stderr, "ERROR: %s: Line %d: Failed to read a texture image file `%s'\n",
				filename, line, name.c_str());
			++err;
		    } else {
			//cv::cvtColor(img, img, CV_BGR2RGB);
			texture = new Texture(name, img);
			scene.textures[name] = texture;
			texture->generate_mipmaps();
		    }
		} 
		if( header == "map_Ka" ) current->ambient_map = texture;
		else if( header == "map_Kd" ) current->diffuse_map = texture;
		else if( header == "map_Ks" ) current->specular_map = texture;
		else current->bump_map = texture;
	    }
	} else if( header == "Km" ) {
	    argErr = !scan_float(p, current->bump_scale);
	} else if( header == "illum" ||
		   header == "d" ||
		   header == "Tr" ||
		   header == "Tf" ||
		   header == "Ke" ||
		   header == "sharpness" ||
		   header == "map_Ns" || 
		   header == "map_d" || 
		   header == "map_D" || 
		   header == "bump" || 
		   header == "disp" || 
		   header == "refl" || 
		   header == "decal" ) {
	    /* ignore */
	    p = &buf[strlen(buf)-1];
	} else {
	    fprintf(stderr, "ERROR: %s: Line %d: Unknown line header `%s'\n", 
		    filename, line, header.c_str());
	    ++err;
	    break;
	}

	if( argErr ) {
	    fprintf(stderr, "ERROR: %s: Line %d: %s: Too few arguments\n", 
		    filename, line, header.c_str());
	    ++err;
	}

	string excess;
	if( scan_string(p, excess) ) {
	    fprintf(stderr, "ERROR: %s: Line %d: %s: Excess argument `%s'\n", 
		    filename, line, header.c_str(), excess.c_str());
	    ++err;
	}
    }

    fclose(fp);
    return err > 0 ? RT_ERROR : 0;
}

/* 面の情報を一時的に保管するための構造体 */
struct face_data {
    vector<ivec3> vertices;   /* v/t/n 形式のインデックス番号が入る 
			         (球の場合は空) */
    Material *material;
    int line;
    face_data() : material(NULL), line(0) {}
    face_data(Material *m, int l) : material(m), line(l) {}
};

struct face_status {
    int  obj_begin, obj_end;
    face_status() : obj_begin(-1) {}
};

int 
RT_ReadObjFile(const InputParameter& param, Scene& scene)
{
    const char *filename = NULL;
    string dirname = getDirName(param.file_name);
    int  err = 0;
    int  line = 0;
    char buf[RT_BUFLEN];
    vector<vector<vec3> > vertices, normals;
    vector<vector<vec2> > texCoords;
    Material *currentMaterial = &Material::default_material;
    vector<vector<face_data> > faceBuffer;
    vector<face_status> faceStatus;
    int nfaces = 0;

    for( int frame = 0; frame <= param.end_frame - param.start_frame; frame++ ) {
	if( filename ) free((void*)filename);
	sprintf(buf, param.obj_file.c_str(), frame + param.start_frame);
	filename = strdup(buf);
	FILE *fp = fopen((dirname + filename).c_str(), "r");
	if( !fp ) {
	    if( frame == 0 ) {
		fprintf(stderr, "ERROR: ");
		perror((dirname + filename).c_str());
		return RT_ERROR;
	    } else {
		continue;
	    }
	}
	fprintf(stderr, "Reading %s...\r", filename);

	if( faceBuffer.size() < frame+1 ) {
	    faceBuffer.resize(frame+1);
	    vertices.resize(frame+1);
	    normals.resize(frame+1);
	    texCoords.resize(frame+1);
	}

	while( fgets(buf, RT_BUFLEN, fp) ) {
	    line++;
	    if( err >= MAX_ERRS )  break;
	    if( strlen(buf) == RT_BUFLEN - 1 ) {
		fprintf(stderr, "ERROR: %s: Line %d: Too long input line\n", 
			filename, line);
		++err;
	    }
	
	    const char *p = buf;
	    string header;
	    bool argErr = false;
	    if( !scan_string(p, header) ) continue;

	    if( header == "mtllib" ) {
		string name;
		if( !(argErr = !scan_string(p, name)) ) {
		    if( RT_ReadMaterialFile(name, dirname,
					    getDirName(dirname + filename), 
					    scene) == RT_ERROR ) {
			++err;
		    }
		}
	    } else if( header == "usemtl" ) {
		string name;
		if( !(argErr = !scan_string(p, name)) ) {
		    std::map<string,Material*>::iterator m = scene.materials.find(name);
		    if( m == scene.materials.end() ) {
			fprintf(stderr, "ERROR: %s: Line %d: %s: No such material name\n",
				filename, line, name.c_str());
			++err;
		    } else {
			currentMaterial = m->second;
		    }
		}
	    } else if( header == "v" ) {
		vec3 v;
		argErr = (!scan_float(p, v.x) ||
			  !scan_float(p, v.y) ||
			  !scan_float(p, v.z));
		if( !argErr ) vertices[frame].push_back(v);
		/* 4番目の要素がある場合は無視する */
		float trash;
		scan_float(p, trash);
	    } else if( header == "vb" ) {
		vec3 v;
		fread((void*)&v, sizeof(vec3), 1, fp) ==0; // ==0 suppresses warning
		vertices[frame].push_back(v);
	    } else if( header == "vn" ) {
		vec3 v;
		argErr = (!scan_float(p, v.x) ||
			  !scan_float(p, v.y) ||
			  !scan_float(p, v.z));
		/* 法線ベクトルはここで正規化する */
		if( !argErr ) normals[frame].push_back(normalize(v));
	    } else if( header == "vnb" ) {
		vec3 v;
		fread((void*)&v, sizeof(vec3), 1, fp) ==0;
		normals[frame].push_back(v);
	    } else if( header == "vt" ) {
		vec2 v;
		argErr = (!scan_float(p, v.s) ||
			  !scan_float(p, v.t));
		if( !argErr ) texCoords[frame].push_back(v);
		/* 3番目の要素がある場合は無視する */
		float trash;
		scan_float(p, trash);
	    } else if( header == "vtb" ) {
		vec2 v;
		fread((void*)&v, sizeof(vec2), 1, fp) ==0;
		texCoords[frame].push_back(v);
	    } else if( header == "f" ) {
		/* 無頂点のfは前フレームからの継続、1頂点のは削除を意味する */
		ivec3 v;
		faceBuffer[frame].push_back(face_data(currentMaterial, line));
		nfaces = std::max<int>(nfaces, faceBuffer[frame].size());
		while( scan_slashed_three_integers(p, v) ) {
		    if( v.x < 0 ) v.x += vertices.size() + 1; 
		    if( v.y < 0 ) v.y += texCoords.size() + 1; 
		    if( v.z < 0 ) v.z += normals.size() + 1; 
		    faceBuffer[frame].back().vertices.push_back(v);
		}
	    } else if( header == "fb" ) {
		faceBuffer[frame].push_back(face_data(currentMaterial, line));
		nfaces = std::max<int>(nfaces, faceBuffer[frame].size());
		int num;
		fread((void*) &num, sizeof(int), 1, fp) ==0; 
		for( int i = 0; i < num; i++ ) {
		    ivec3 v;
		    fread((void*)&v, sizeof(ivec3), 1, fp) ==0;
		    if( v.x < 0 ) v.x += vertices.size() + 1; 
		    if( v.y < 0 ) v.y += texCoords.size() + 1; 
		    if( v.z < 0 ) v.z += normals.size() + 1; 
		    faceBuffer[frame].back().vertices.push_back(v);
		}
	    } else if( header == "sphere" ) {
		fprintf(stderr, "sphere is no longer supported\n");
	    } else if( header == "o" ||
		       header == "g" ||
		       header == "s" ||
		       header == "vp" ) {
		/* ignore */
		p = &buf[strlen(buf)-1];
	    } else {
		fprintf(stderr, "ERROR: %s: Line %d: Unknown line header `%s'\n", 
			filename, line, header.c_str());
		++err;
		break;
	    }
	    
	    if( argErr ) {
		fprintf(stderr, "ERROR: %s: Line %d: %s: Too few arguments\n", 
			filename, line, header.c_str());
		++err;
	    }
	    
	    string excess;
	    if( scan_string(p, excess) ) {
		fprintf(stderr, "ERROR: %s: Line %d: %s: Excess argument `%s'\n", 
			filename, line, header.c_str(), excess.c_str());
		++err;
	    }
	}

	fclose(fp);
    }
    fprintf(stderr, "\n");

    /* 欠けているフレームをスプライン補間する */
    if( faceBuffer.size() > 1 ) fprintf(stderr, "Interpolating frames...\n");
    for( int fid = 0; fid < nfaces; fid++ ) {
	vector<int>  frames;  // キーフレームの番号のリスト
	for( int frame = 0; frame < faceBuffer.size(); frame++ ) {
	    if( faceBuffer[frame].size() <= fid ) continue;
	    face_data *f = &faceBuffer[frame][fid];
	    if( f->vertices.size() == 0 )  continue;
	    if( f->vertices.size() >= 3 ) frames.push_back(frame);

	    if( frames.size() > 0 &&
		(frame == faceBuffer.size() - 1 ||  // 最後のフレームのとき
		 f->vertices.size() < 3) ) {   // 途中で物体が削除されたとき
		face_data *f0 = &faceBuffer[frames[0]][fid];

		/* 頂点座標に全く変化がない場合は、最初のものを除いて
		   無駄な face設定を削除する （法線やテキスチャ座標だけが
		   変わるのは、今のところ無駄な更新と見なす) */
		/* GBVHでない場合でも、変化がにものはface設定を削除する（それによって
		   三角形にしたときには、第１フレームでの三角形への参照が各フレームに
		   追加される）。
		   以前は、GBVHでない場合は、不動の物体でもspline補間していた */
		bool no_motion = true;
		for( int k = 1; no_motion && k < frames.size(); k++ ) {
		    face_data *f1 = &faceBuffer[frames[k]][fid];
		    if( f0->vertices.size() != f1->vertices.size() ) {
			no_motion = false;
		    } else {
			for( int i = 0; i < f0->vertices.size(); i++ ) {
			    if( vertices[frames[0]][f0->vertices[i][0]-1] !=
				vertices[frames[k]][f1->vertices[i][0]-1] ) {
				no_motion = false;
				break;
			    }
			}
		    }
		}

		if( no_motion ) {
		    for( int k = 1; k < frames.size(); k++ ) {
			faceBuffer[frames[k]][fid].vertices.clear();
		    }
		} else if( frames.back() - frames.front() >= frames.size() ) {
		    /* スプライン補間 */
		    int nv = f0->vertices.size();
		    for( int i = 0; i < nv; i++ ) { 
			vector<vec3>  av, an;
			vector<vec2>  at;
			vector<float>  times;
			for( int k = 0; k < frames.size(); k++ ) {
			    face_data *f1 = &faceBuffer[frames[k]][fid];
			    if( f1->vertices.size() != nv ) {
				fprintf(stderr, "No. of face vertices mismatch\n (face_id=%d frame=%d)\n", fid, frames[k]);
				++err;
			    } else {
				ivec3 vtn = f1->vertices[i];
				av.push_back(vertices[frames[k]][vtn[0]-1]);
				if( vtn[1] ) at.push_back(texCoords[frames[k]][vtn[1]-1]);
				if( vtn[2] ) an.push_back(normals[frames[k]][vtn[2]-1]);
			    }
			    times.push_back(frames[k]);
			}
			if( err ) continue;

			Spline<vec3>  spv(av, times), spn(an, times);
			Spline<vec2>  spt(at, times);
			for( int fr = frames.front(); fr < frames.back(); fr++ ) {
			    if( faceBuffer[fr].size() < fid+1 ) {
				faceBuffer[fr].resize(fid+1);
			    }
			    if( faceBuffer[fr][fid].vertices.size() == nv ) {
				/* キーフレームの場合 */
				continue;
			    }
			    vertices[fr].push_back(spv.get_value(fr));
			    if( spt.is_valid() ) texCoords[fr].push_back(spt.get_value(fr));
			    if( spn.is_valid() ) normals[fr].push_back(spn.get_value(fr));
			    faceBuffer[fr][fid].vertices.push_back(
				ivec3(vertices[fr].size(),
				      spt.is_valid() ? texCoords[fr].size() : 0,
				      spn.is_valid() ? normals[fr].size() : 0));
			    faceBuffer[fr][fid].material = f0->material;
			}
		    }
		}

		frames.clear();
	    }
	}
    }
    
    /* faceBuffer に格納されている番号リストから三角形を生成し、
       scenario に登録する */
    fprintf(stderr, "Generating triangles...\n");
    scene.scenario.resize(scene.view_scenario.size());
    faceStatus.resize(nfaces);
    for( int frame = 0; frame < faceBuffer.size(); frame++ ) {
	if( scene.scenario.size() < frame+1 ) {
	    scene.scenario.resize(frame+1);
	    scene.view_scenario.resize(frame+1);
	}

	for( int fid = 0; fid < nfaces; fid++ ) {
	    face_data *f = &faceBuffer[frame][fid];
	    if( err >= MAX_ERRS )  break;
	    if( fid >= faceBuffer[frame].size() || f->vertices.size() == 0 ) { /* 保持 */
		if( !(param.build_type == BUILD_TREE_GBVH || DO_REFIT) ) {
		    /* GBVH や REFITでない場合、前のフレームの三角形を再登録 */
		    if( frame != 0 && faceStatus[fid].obj_begin != -1 ) {
			for( int k = faceStatus[fid].obj_begin;
			     k < faceStatus[fid].obj_end; k++ ) {
			    scene.scenario[frame].push_back(k);
			}
		    }
		}
		continue;
	    }

	    /* 前のフレームの三角形を削除 */
	    if( param.build_type == BUILD_TREE_GBVH ) {
		if( frame != 0 && faceStatus[fid].obj_begin != -1 ) {
		    for( int k = faceStatus[fid].obj_begin;
			 k < faceStatus[fid].obj_end; k++ ) {
#if USE_EXPIRE
			scene.objects[k]->expire = frame;
#else
			scene.scenario[frame].push_back(~k);
#endif
			//printf("frame=%d delete %d\n", frame, k);
		    }
		}
	    }
#if DO_REFIT
	    int del_begin = faceStatus[fid].obj_begin;
	    if( frame != 0 && faceStatus[fid].obj_end - del_begin != f->vertices.size() - 2 ) {
		fprintf(stderr, "number of vertices in face %d mismatches\n", fid);
		exit(1);
	    }
#endif

	    if( f->vertices.size() < 3 ) { /* 削除 */
		faceStatus[fid].obj_begin = -1;
		continue;
	    }
	
	    /* ポリゴンは凸だと仮定して、単純な三角形分割を行う */
	    faceStatus[fid].obj_begin = scene.objects.size();
	    for( int i = 1; i < f->vertices.size() - 1; i++ ) {
#if DO_REFIT
		if( frame != 0 && del_begin != -1 ) {
		    scene.scenario[frame].push_back(~(del_begin + (i-1)));
		}
#endif
		int iv0 = f->vertices[0][0];
		int iv1 = f->vertices[i][0];
		int iv2 = f->vertices[i+1][0];
		if( iv0 <= 0 || iv0 > vertices[frame].size() ||
		    iv1 <= 0 || iv1 > vertices[frame].size() ||
		    iv2 <= 0 || iv2 > vertices[frame].size() ) {
		    fprintf(stderr, "ERROR: %s: Line %d: Vertex index out of range\n",
			    filename, f->line);
		    ++err;
		    break;
		}
		Triangle *tr = new Triangle(vertices[frame][iv0 - 1],
					    vertices[frame][iv1 - 1],
					    vertices[frame][iv2 - 1]);
		tr->set_material(f->material);
		scene.scenario[frame].push_back(Action(scene.objects.size()));
		//printf("frame=%d insert %d\n", frame, scene.objects.size());
		scene.objects.push_back(tr);
		
		int it0 = f->vertices[0][1];
		int it1 = f->vertices[i][1];
		int it2 = f->vertices[i+1][1];
		if( it0 || it1 || it2 ) {
		    if( it0 <= 0 || it0 > texCoords[frame].size() ||
			it1 <= 0 || it1 > texCoords[frame].size() ||
			it2 <= 0 || it2 > texCoords[frame].size() ) {
			fprintf(stderr, "ERROR: %s: Line %d: Texture-coordinate index out of range\n",
				filename, f->line);
			++err;
			break;
		    }
		    tr->set_texture_coord(&texCoords[frame][it0 - 1],
				    &texCoords[frame][it1 - 1],
				    &texCoords[frame][it2 - 1]);
		}
		
		int in0 = f->vertices[0][2];
		int in1 = f->vertices[i][2];
		int in2 = f->vertices[i+1][2];
		if( in0 || in1 || in2 ) {
		    if( in0 <= 0 || in0 > normals[frame].size() ||
			in1 <= 0 || in1 > normals[frame].size() ||
			in2 <= 0 || in2 > normals[frame].size() ) {
			fprintf(stderr, "ERROR: %s: Line %d: Normal-vector index out of range\n",
				filename, f->line);
			++err;
			break;
		    }
		    tr->set_normal(&normals[frame][in0 - 1],
				  &normals[frame][in1 - 1],
				  &normals[frame][in2 - 1]);
		}
	    }
	    faceStatus[fid].obj_end = scene.objects.size();
	}

	/* free unnecessary data */
	vector<vec3>().swap(vertices[frame]);
	vector<vec3>().swap(normals[frame]);
	vector<vec2>().swap(texCoords[frame]);
	vector<face_data>().swap(faceBuffer[frame]);
    }

    return err > 0 ? RT_ERROR : 0;
}
